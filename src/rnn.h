#ifndef CE_RNN_H
#define CE_RNN_H
#include <stddef.h>
#include <stdint.h>
#define RNN_MAX_H 128
#define RNN_MAX_V 96
#define RNN_HEADER 24
/* Model stays in read-only storage; only recurrent state/logits consume RAM. */
typedef struct {
    uint16_t h, v;
    const uint8_t *vocab;
    const int8_t *wx, *wh, *bh, *wy, *by;
    int8_t state[RNN_MAX_H];
    int32_t logits[RNN_MAX_V];
} rnn_t;
const char *rnn_load(rnn_t *model, const uint8_t *bytes, size_t length);
void rnn_reset(rnn_t *model);
int rnn_token(const rnn_t *model, char c);
int rnn_step(rnn_t *model, unsigned token);
uint32_t rnn_macs(const rnn_t *model);
#endif
