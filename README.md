# CE AI — offline calculator neural experiment

A tiny character-level recurrent neural network that generates text **on a TI-84 Plus CE**. Training happens locally on a computer; after transferring the program and model, inference needs no computer, internet, account, or API key.

This first prototype includes a NumPy trainer, integer inference engine, Python reference implementation, model exporter, storage/operation inspector, desktop runner, tests, and a compiled calculator interface with a benchmark screen.

**Status:** native calculator compilation and desktop numerical tests pass. The included model gives coherent replies to several prompts represented in its tiny training corpus. **Physical calculator execution, keypad/display behavior, and calculator speed remain unverified.**

## Try the calculator build

The generated `outputs/CE-AI-starter.zip` contains:

- `CEAI.8xp`: the native calculator program.
- `demo.8xv`: the 96-hidden-unit model, installed as AppVar **CEAIMDL**.
- `clibs.8xg`: official CE libraries v15.0 (redistribution license included).
- `INSTALL.txt` and `LICENSE`: installation, controls, and limitations.

1. Have a compatible native C/assembly launcher already working on the calculator. Native programs on OS 5.5+ need an appropriate launcher workaround; this package does not install one or change the OS.
2. Transfer the three calculator files using TI Connect CE or a compatible transfer tool. Prefer archive storage. CE Link accepts these file types, although its physical USB transfer is also still unverified.
3. Launch **CEAI**. It reads **CEAIMDL** without changing other variables.
4. Press **GRAPH** for the benchmark. Photograph the result screen to capture neural steps, elapsed clock ticks, ticks/second, multiply-accumulate operations per step, OS free RAM, and the model state struct size. Desktop timing is not an estimate of calculator speed.
5. Press **2nd** to generate from the initial `hello` prompt. Press **ENTER** to edit prompts with the on-screen keyboard: arrows select, ENTER adds a character, DEL erases, 2nd finishes editing. The keyboard contains only characters supported by the loaded model.
6. **CLEAR** stops prompt processing/generation between neural steps; from the main screen it exits. One slow neural step must finish before cancellation is checked.

Try `hello`, `who are you`, `what is a model`, `what is gravity`, and `tell me a joke`. The app wraps the text as `user: ...\nbot: ` and stops at the first generated newline or 240 characters. It resets neural state for each request; it does not yet remember previous conversations or learn on the calculator.

Only one model is loaded at a time. Exported models default to the same CEAIMDL variable name, so transferring another model replaces that model variable. The program opens it read-only and does not create conversation/settings files.

## What this model actually does

The included original corpus has 39 short question/answer examples, repeated in shuffled order. Its facts and replies were written for this project; no private messages, notes, calculator exports, or downloaded datasets are used.

The program does **not** look up canned responses: each generated character comes from neural matrix operations and learned weights. However, this tiny training set makes the result mostly a memorization demonstration. Unfamiliar prompts can produce nonsense. It does not provide general reasoning, reliable mathematics, or a ChatGPT-level assistant.

Measured desktop integer output from the supplied model:

```text
user: hello
bot: hello! i am a tiny calculator bot.

user: who are you
bot: i am ce ai, a tiny neural experiment.

user: what is a model
bot: a model is a set of learned numerical weights.

user: what is love
bot: a bot: numbercusatlo
```

The train/validation split holds out a tail of the same repeated corpus. The metrics in `models/demo.json` are next-character metrics on that corpus and **do not measure generalization to unseen questions**.

## Model sizes

| Included model | Hidden state | Model bytes | Multiply-accumulates per step | Purpose |
| --- | ---: | ---: | ---: | --- |
| `models/tiny64.*` | 64 | 8,474 | 6,208 | Smaller baseline; less coherent |
| `models/demo.*` | 96 | 15,738 | 12,384 | Default trained demonstration |

Both use 33 characters. Weights are signed 8-bit fixed-point values. The engine points directly into read-only model storage; it does not copy the full model into RAM. Recurrent state, output scores, screen buffers, runtime, stack, and other app data still use RAM. The on-device benchmark reports actual OS free RAM and `sizeof` the model structure; neither is a claim that all remaining RAM is available as one allocation.

The current file format supports at most 128 hidden units and 96 characters, in a single AppVar. Those are implementation limits, not a promise that larger models would produce better answers.

## Train locally

Python 3.12+ is required for the pinned NumPy version. The pinned version was tested under Python 3.15 on macOS.

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python tools/doctor.py
.venv/bin/python tools/train.py
```

Defaults reproduce the larger model configuration: 96 hidden units, 18,000 updates, 96-character training sequences, seed 42. Reports include corpus hash, settings, library version, losses, and model size. Floating-point differences between NumPy/platforms can affect retraining; the exported integer model has a specified, tested rounding rule.

For your own corpus, use lowercase printable ASCII and newlines, ideally in this format:

```text
user: hello
bot: hello there!

user: who are you
bot: i am a tiny calculator experiment.
```

```sh
.venv/bin/python tools/train.py --data data/my-corpus.txt --output models/custom --hidden 96 --steps 18000 --sequence 96
.venv/bin/python tools/model.py models/custom.bin --prompt 'user: hello
bot: ' --count 120
.venv/bin/python tools/model_info.py models/custom.bin
```

The trainer lowercases text and validates the alphabet. The calculator's prompt wrapper needs the characters in `user: \nbot: `; keep those in custom training corpora to use the existing UI. A small `.npz` checkpoint retains float weights for re-export; optimizer state is not saved, and training resume is not implemented.

Export a checkpoint without retraining:

```sh
.venv/bin/python tools/export.py models/demo.npz --output models/reexport --name CEAIMDL
```

All output models use that same variable name unless specified. A custom name needs a corresponding change in `src/main.c`.

## Build and verify

The [CE C Toolchain v15.0](https://github.com/CE-Programming/toolchain/releases/tag/v15.0) is required for the calculator program. Set `CEDEV` to the installed toolchain directory, or place its `bin/` on PATH:

```sh
CEDEV=/path/to/CEdev tools/build-calculator.sh
```

This produces `bin/CEAI.8xp`. The compiler can emit upstream assembler linker warnings about `.note.GNU-stack`; the source compiles without C warnings.

The desktop runner uses the **same C inference engine**:

```sh
mkdir -p outputs
clang -std=c11 -Wall -Wextra -Werror -O2 src/rnn.c tools/desktop.c -o outputs/ceai-desktop
outputs/ceai-desktop models/demo.bin 'user: hello
bot: ' 120
.venv/bin/python tests/check.py
```

Tests compile the C engine with AddressSanitizer and UndefinedBehaviorSanitizer, compare every hidden state and output score against Python across multiple model dimensions including the maximum supported size, compare trained-model generation, reject truncated/corrupt models, check TI file framing/checksums, and smoke-test training/checkpoint export. This is desktop verification, not emulator or hardware execution.

To fetch the pinned official runtime libraries and their license, then package the build:

```sh
.venv/bin/python tools/fetch_libraries.py
.venv/bin/python tools/package.py
```

The fetcher verifies SHA-256 hashes before saving. `tools/package.py` creates the starter ZIP and source toolkit ZIP, verifies TI checksums, and records SHA-256 hashes. Toolchains and virtual environments are excluded from the packages.

## Files and format

- `src/rnn.c`, `src/rnn.h`: portable integer inference and defensive model loader.
- `src/main.c`: calculator keyboard, generation UI, benchmark.
- `tools/train.py`: local hard-tanh RNN trainer using backpropagation through time and AdaGrad.
- `tools/model.py`: binary model format, exact integer reference, TI AppVar framing.
- `tools/export.py`, `tools/model_info.py`: export and inspect.
- `tools/desktop.c`: same engine, terminal output and numerical traces.
- `models/*.npz`, `*.bin`, `*.8xv`, `*.json`: float checkpoints, integer models, calculator files, training reports.
- `data/demo.txt`, `tools/make_demo.py`: original toy corpus and deterministic generator.
- `FORMAT.md`: byte layout and arithmetic rules.

No paid services, API credentials, external model downloads, or ROM images are needed. Source, original corpus, and included trained model files are MIT licensed. Official CE libraries retain their own BSD license. This project is independent of Texas Instruments.

References: [fileioc read-only AppVar access](https://ce-programming.github.io/toolchain/libraries/fileioc.html), [CE toolchain](https://ce-programming.github.io/toolchain/), [official CE libraries v15.0](https://github.com/CE-Programming/libraries/releases/tag/v15.0).
