#include "../src/rnn.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
static rnn_t model;
static int feed(unsigned token, int trace) {
    int next = rnn_step(&model, token);
    if (trace) {
        printf("[%u,[", token);
        for (unsigned i = 0; i < model.h; ++i) printf(i ? ",%d" : "%d", model.state[i]);
        printf("],[");
        for (unsigned i = 0; i < model.v; ++i) printf(i ? ",%ld" : "%ld", (long)model.logits[i]);
        puts("]]");
    }
    return next;
}
int main(int argc, char **argv) {
    uint8_t *bytes; long size; FILE *file; const char *error;
    int token = 0, trace = argc > 4 && !strcmp(argv[4], "--trace");
    unsigned count; clock_t start;
    if (argc < 4) { fprintf(stderr, "Usage: ceai MODEL.bin PROMPT COUNT [--trace]\n"); return 2; }
    count = (unsigned)strtoul(argv[3], NULL, 10);
    if (count > 100000) return 2;
    file = fopen(argv[1], "rb"); if (!file) return 2;
    if (fseek(file, 0, SEEK_END) || (size = ftell(file)) < 0 || size > 65512) { fclose(file); return 2; }
    rewind(file); bytes = malloc((size_t)size + 1);
    if (!bytes) { fclose(file); return 2; }
    if (fread(bytes, 1, (size_t)size, file) != (size_t)size) { fclose(file); free(bytes); return 2; }
    fclose(file); error = rnn_load(&model, bytes, (size_t)size);
    if (error) { fprintf(stderr, "%s\n", error); free(bytes); return 1; }
    if (!*argv[2]) { fprintf(stderr, "Prompt must not be empty\n"); free(bytes); return 2; }
    start = clock();
    for (const char *p = argv[2]; *p; ++p) {
        int input = rnn_token(&model, *p);
        if (input < 0) { fprintf(stderr, "Character not in model alphabet\n"); free(bytes); return 2; }
        token = feed((unsigned)input, trace);
    }
    for (unsigned i = 0; i < count; ++i) {
        if (!trace) putchar(model.vocab[token]);
        token = feed((unsigned)token, trace);
    }
    if (!trace) putchar('\n');
    fprintf(stderr, "Desktop only: %u generated chars, %.6f CPU seconds, %lu MACs/step; state struct %zu bytes\n", count,
        (double)(clock() - start) / CLOCKS_PER_SEC, (unsigned long)rnn_macs(&model), sizeof(model));
    free(bytes); return 0;
}
