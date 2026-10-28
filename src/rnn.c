#include "rnn.h"
#include <string.h>
static uint16_t le16(const uint8_t *p) { return (uint16_t)p[0] | (uint16_t)p[1] << 8; }
static uint32_t le32(const uint8_t *p) {
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}
const char *rnn_load(rnn_t *m, const uint8_t *b, size_t n) {
    uint16_t h, v, i, j;
    uint32_t size, hash = UINT32_C(2166136261);
    const int8_t *p;
    memset(m, 0, sizeof(*m));
    if (!b || n < RNN_HEADER || memcmp(b, "CERNN01\0", 8)) return "Invalid model signature";
    h = le16(b + 8); v = le16(b + 10);
    if (!h || h > RNN_MAX_H || v < 2 || v > RNN_MAX_V) return "Unsupported dimensions";
    if (le32(b + 20)) return "Unsupported format flags";
    size = (uint32_t)v + (uint32_t)h * v * 2 + (uint32_t)h * h + h + v;
    if (le32(b + 12) != size || size + RNN_HEADER != n || n > 65512) return "Invalid model length";
    for (i = 0; i < v; ++i) {
        if (b[RNN_HEADER + i] != '\n' && (b[RNN_HEADER + i] < 32 || b[RNN_HEADER + i] > 126)) return "Invalid alphabet";
        for (j = 0; j < i; ++j) if (b[RNN_HEADER + i] == b[RNN_HEADER + j]) return "Duplicate alphabet character";
    }
    for (size_t k = RNN_HEADER; k < n; ++k) hash = (hash ^ b[k]) * UINT32_C(16777619);
    if (hash != le32(b + 16)) return "Model checksum failed";
    m->h = h; m->v = v; m->vocab = b + RNN_HEADER;
    p = (const int8_t *)(m->vocab + v);
    m->wx = p; p += (size_t)h * v;
    m->wh = p; p += (size_t)h * h;
    m->bh = p; p += h;
    m->wy = p; p += (size_t)v * h;
    m->by = p;
    return NULL;
}
void rnn_reset(rnn_t *m) { memset(m->state, 0, sizeof(m->state)); memset(m->logits, 0, sizeof(m->logits)); }
int rnn_token(const rnn_t *m, char c) {
    for (unsigned i = 0; i < m->v; ++i) if (m->vocab[i] == (uint8_t)c) return (int)i;
    return -1;
}
int rnn_step(rnn_t *m, unsigned token) {
    int8_t next[RNN_MAX_H];
    int best = 0;
    if (token >= m->v || !m->h || !m->v) return -1;
    for (unsigned i = 0; i < m->h; ++i) {
        int32_t sum = ((int32_t)m->wx[(size_t)i * m->v + token] + m->bh[i]) * 127;
        for (unsigned j = 0; j < m->h; ++j) sum += (int32_t)m->wh[(size_t)i * m->h + j] * m->state[j];
        sum = sum >= 0 ? (sum + 32) / 64 : -((-sum + 32) / 64);
        next[i] = (int8_t)(sum > 127 ? 127 : sum < -127 ? -127 : sum);
    }
    memcpy(m->state, next, m->h);
    for (unsigned i = 0; i < m->v; ++i) {
        int32_t sum = (int32_t)m->by[i] * 127;
        for (unsigned j = 0; j < m->h; ++j) sum += (int32_t)m->wy[(size_t)i * m->h + j] * next[j];
        m->logits[i] = sum;
        if (sum > m->logits[best]) best = (int)i;
    }
    return best;
}
uint32_t rnn_macs(const rnn_t *m) { return (uint32_t)m->h * m->h + (uint32_t)m->v * m->h; }
