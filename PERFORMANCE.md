# Performance pass — 2026-10-28

The model was already quantized: signed 8-bit Q6 weights and Q7 recurrent state, with integer-only inference. This update optimizes the runtime without changing the default model or its output.

## Implemented

- Native signed 24-bit accumulators on eZ80. With H<=128, signed byte weights, and state magnitude <=127, every accumulator is bounded by 2,113,280 in magnitude, below the signed 24-bit limit of 8,388,607. Desktop builds retain 32-bit accumulators.
- Exact signed 16-bit products, sequential weight pointers, and saturation before shift-based normalization. Integer rounding and tie behavior match the Python reference.
- Speed-oriented `-O2` compilation instead of `-Oz`. The larger program is a deliberate code-size tradeoff.
- Stock **48 MHz** CPU mode through the documented `boot_Set48MHzModeI` routine. This is normal full speed, not an above-specification overclock.
- Screen refresh is capped at about eight redraws per second during generation instead of redrawing after every generated character. If individual steps are slower than that, each completed step may still redraw.
- **TRACE** toggles an optional full model copy in heap RAM. If allocation fails, direct AppVar reads remain active. The original model AppVar is opened read-only in both modes. The copy is freed on toggle or exit.
- **GRAPH** measures 128 neural steps at the current memory placement. It reports raw clock ticks and clock frequency so timing remains interpretable. Run once with direct archived weights, then TRACE and GRAPH again to compare RAM. A RAM copy is not assumed to be faster.

The default app's model file remains 15,738 bytes. The new calculator program is 18,498 bytes, versus 14,289 bytes in the first size-oriented build.

## Verified evidence

Eight desktop tests pass, including sanitizer checks, exact C/Python state/output-score parity at five dimension combinations, maximum-accumulator cases, and unchanged default 8-bit export. The same engine is compiled into the calculator app.

In the emitted CE target `rnn_step` assembly:

| Structural check | Original | Updated |
| --- | ---: | ---: |
| General 32-bit multiply helper call sites | 4 | 0 |
| General 32-bit addition helper call sites | 4 | 0 |
| Assembly source lines in function | 564 | 466 |

These are static code-generation observations. They do not measure instruction cycles or establish a calculator speedup. The updated loop still uses the compiler's native-width multiply helper. A hand-written assembly kernel remains a possible future experiment; it is not included or hardware-verified here.

To check a rebuilt target:

```sh
.venv/bin/python tools/check_codegen.py obj/lto.s
```

Physical calculator execution, memory-copy allocation, UI behavior, and actual performance are still unverified. No clock setting was applied to a connected calculator during development.

## Lower-bit quantization experiment

The quantization tool tests symmetric 4-, 6-, and 8-bit coefficient grids. Runtime files still store each weight in a byte; this experiment evaluates quality without pretending that nibble packing speeds up the CPU. There is no packed 4-bit backend.

```sh
.venv/bin/python tools/quantize.py models/demo.npz --compare
```

On the supplied repeated toy corpus, direct post-training reduction damaged familiar replies:

| Coefficient grid | Reply to `hello` |
| --- | --- |
| 4-bit equivalent | `e  how` |
| 6-bit equivalent | `gravity frightrbins.` |
| 8-bit | `hello! i am a tiny calculator bot.` |

The 8-bit model is therefore retained. Lower-bit models would need further training and a measured runtime benefit to justify their use. These samples and the same-domain next-character metrics are not general conversational benchmarks.

## CPU, VRAM and memory timing

The CE toolchain exposes 6 MHz and 48 MHz CPU routines. It does not provide a supported above-48-MHz overclock API. This build selects the normal 48 MHz mode and does not alter oscillator hardware, voltage, flash wait-state registers, LCD timing registers, or memory-controller settings.

VRAM is memory used by the display controller, which also consumes memory bandwidth through DMA. It is not a separate GPU with a supported independent VRAM overclock setting exposed by this toolchain. Graphx already uses an 8-bit display mode, reducing display traffic relative to 16-bit modes. Reduced redraws and better access patterns are practical software optimizations.

Flash behavior depends on hardware revision: newer serial-flash calculators use a cache, so contiguous reads can be fast. Copying everything to RAM can therefore help or hurt. Benchmarking both placements is more informative than assuming RAM is always faster.

Sources checked 2026-10-28: [CE clock control API](https://ce-programming.github.io/toolchain/headers/sys/power.html), [documented CE memory wait states, serial-flash cache and LCD DMA observations](https://wikiti.brandonw.net/index.php?title=84PCE:Wait_States), and locally inspected v15.0 `sys/power.h`, `sys/lcd.h` and emitted assembly. WikiTI describes developer hardware measurements, not a manufacturer performance guarantee.
