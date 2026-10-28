/* CE AI experiment: local integer neural inference. MIT; see LICENSE. */
#include <tice.h>
#include <graphx.h>
#include <fileioc.h>
#include <keypadc.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "rnn.h"
static rnn_t model;
static char prompt[97] = "hello", reply[241], status[80];
static char keys[RNN_MAX_V + 1];
static unsigned key_count;
static unsigned cursor;
static size_t free_ram;
static uint16_t model_size;
static uint8_t handle;
static uint8_t key(void) { uint8_t k; do { k = os_GetCSC(); } while (!k); return k; }
static void line(const char *text, unsigned row) { gfx_PrintStringXY(text, 8, (int)(row * 10 + 6)); }
static void wrapped(const char *s, unsigned row, unsigned max_rows) {
    char buffer[39]; unsigned used = 0, lines = 0;
    for (; *s && lines < max_rows; ++s) {
        if (*s != '\n') buffer[used++] = *s;
        if (*s == '\n' || used == 38) { buffer[used] = 0; line(buffer, row + lines++); used = 0; }
    }
    if (used && lines < max_rows) { buffer[used] = 0; line(buffer, row + lines); }
}
static void screen(void) {
    char buffer[80];
    gfx_FillScreen(255); gfx_SetTextFGColor(0); gfx_SetTextBGColor(255);
    line("CE AI / OFFLINE NEURAL EXPERIMENT", 0);
    snprintf(buffer, sizeof(buffer), "%u hidden / %u chars / %u bytes", model.h, model.v, model_size); line(buffer, 1);
    line("Prompt:", 3); wrapped(prompt, 4, 3);
    line("Reply:", 8); wrapped(reply, 9, 7);
    wrapped(status, 17, 2);
    line("ENTER type   2nd generate", 20);
    line("GRAPH benchmark   CLEAR quit", 21);
    gfx_BlitBuffer();
}
static void keyboard(void) {
    for (;;) {
        size_t length = strlen(prompt);
        gfx_FillScreen(255); line("EDIT PROMPT (lowercase)", 0); wrapped(prompt, 2, 3);
        unsigned first = (cursor / 35) * 35;
        for (unsigned i = first; i < key_count && i < first + 35; ++i) {
            char cell[4] = {' ', keys[i], ' ', 0};
            unsigned x = 10 + ((i - first) % 7) * 43, y = 75 + ((i - first) / 7) * 17;
            if (i == cursor) { gfx_SetColor(24); gfx_FillRectangle(x, y - 2, 30, 13); }
            gfx_PrintStringXY(cell, (int)x, (int)y);
        }
        line("Arrows select / ENTER adds letter", 20);
        line("DEL erase / 2nd done / CLEAR done", 21);
        gfx_BlitBuffer();
        switch (key()) {
            case sk_Left: cursor = (cursor + key_count - 1) % (key_count); break;
            case sk_Right: cursor = (cursor + 1) % (key_count); break;
            case sk_Up: if (cursor >= 7) cursor -= 7; break;
            case sk_Down: if (cursor + 7 < key_count) cursor += 7; break;
            case sk_Del: if (length) prompt[length - 1] = 0; break;
            case sk_Enter: if (length < sizeof(prompt) - 1) { prompt[length] = keys[cursor]; prompt[length + 1] = 0; } break;
            case sk_Clear: case sk_2nd: return;
        }
    }
}
static bool cancelled(void) { kb_Scan(); return (kb_Data[6] & kb_Clear) != 0; }
static bool feed_text(const char *text, int *next) {
    for (; *text; ++text) {
        int token = rnn_token(&model, *text);
        if (token < 0) { snprintf(status, sizeof(status), "Unsupported character code %u", (unsigned)(uint8_t)*text); return false; }
        if (cancelled()) { strcpy(status, "Stopped. No calculator data changed."); return false; }
        *next = rnn_step(&model, (unsigned)token);
    }
    return true;
}
static void generate(void) {
    unsigned count = 0; int next = 0; clock_t start, elapsed;
    rnn_reset(&model); reply[0] = 0;
    strcpy(status, "Processing prompt... CLEAR stops."); screen(); start = clock();
    if (!feed_text("user: ", &next) || !feed_text(prompt, &next) || !feed_text("\nbot: ", &next)) return;
    for (; count < 240; ++count) {
        if (cancelled()) break;
        char c = (char)model.vocab[next];
        /* A newline ends a single toy reply. No canned answers are used. */
        if (c == '\n') break;
        reply[count] = c; reply[count + 1] = 0;
        next = rnn_step(&model, (unsigned)next);
        screen();
    }
    elapsed = clock() - start;
    snprintf(status, sizeof(status), "%u chars / %lu ms incl. prompt+UI", count,
        (unsigned long)((elapsed / CLOCKS_PER_SEC) * 1000 + (elapsed % CLOCKS_PER_SEC) * 1000 / CLOCKS_PER_SEC));
}
static void benchmark(void) {
    const unsigned steps = 128; unsigned done = 0; int next = 0;
    clock_t start, elapsed; void *unused;
    free_ram = os_MemChk(&unused);
    rnn_reset(&model); reply[0] = 0;
    strcpy(status, "Timing 128 neural steps. CLEAR stops."); screen();
    start = clock();
    for (; done < steps; ++done) {
        if (cancelled()) break;
        next = rnn_step(&model, (unsigned)next);
    }
    elapsed = clock() - start;
    snprintf(reply, sizeof(reply), "Neural steps: %u\nElapsed clock ticks: %lu\nClock ticks/second: %lu\nMACs per step: %lu\nOS free RAM: %u bytes\nModel state struct: %u bytes\nModel stays in read-only storage.",
        done, (unsigned long)elapsed, (unsigned long)CLOCKS_PER_SEC, (unsigned long)rnn_macs(&model),
        (unsigned)free_ram, (unsigned)sizeof(model));
    strcpy(status, "Photograph these results for sizing.");
}
int main(void) {
    const char *error; void *unused;
    handle = ti_Open("CEAIMDL", "r");
    gfx_Begin(); gfx_SetDrawBuffer();
    if (!handle) {
        gfx_FillScreen(255); line("Model CEAIMDL not found.", 3);
        line("Send demo.8xv before launching CEAI.", 5); line("Press any key to exit.", 7);
        gfx_BlitBuffer(); key(); gfx_End(); return 1;
    }
    model_size = ti_GetSize(handle);
    error = rnn_load(&model, ti_GetDataPtr(handle), model_size);
    if (error) {
        gfx_FillScreen(255); line(error, 3); line("Press any key to exit.", 5);
        gfx_BlitBuffer(); key(); ti_Close(handle); gfx_End(); return 1;
    }
    free_ram = os_MemChk(&unused);
    for (unsigned i = 0; i < model.v; ++i) if (model.vocab[i] != '\n') keys[key_count++] = (char)model.vocab[i];
    if (rnn_token(&model, 'h') < 0 || rnn_token(&model, 'e') < 0 || rnn_token(&model, 'l') < 0 || rnn_token(&model, 'o') < 0) prompt[0] = 0;
    strcpy(status, "Tiny trained model. Expect mistakes.");
    for (;;) {
        screen();
        switch (key()) {
            case sk_Enter: keyboard(); break;
            case sk_2nd: generate(); break;
            case sk_Graph: benchmark(); break;
            case sk_Clear: ti_Close(handle); gfx_End(); return 0;
        }
    }
}
