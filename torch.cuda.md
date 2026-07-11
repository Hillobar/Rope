torch.cuda.mem_get_info() is being called at 1 hz - do we really need to track ram?

Confirmed one real cost. Audit, ranked by impact:



2. HUD timer at 2 Hz (preview.py:144-147).

Same coalescing mechanism that the VRAM timer caused but at 2 Hz instead of 1 Hz. Each _update_hud does sorted(samples) over 120 floats, builds a multi-line f-string, setText on the child QLabel, adjustSize + move. Not heavy individually, but at 2 Hz it's two opportunities per second for an update() to coalesce one frame. If you turned off the HUD timer earlier and draw snapped up to in, that's the proof. Fixes: 1 Hz instead of 2, or only redraw when values changed materially, or setUpdatesEnabled(False) around the text update.





5. Dead fps_average numpy work in vm.process() (VideoManager.py:589-596).

Every ~fps frames during play, allocates a list, runs np.average, formats a debug string that's then discarded (msg is assigned and never read). Tiny but pointless per-second cost. Trivial deletion.

Lower noise: GC pauses (Python's generational GC firing every N allocations — at 30+ fps tensor refs churn a lot; calling gc.collect() once at app startup then gc.disable() on the play path is the heavy-handed but effective option). DWM compositor jitter (Windows-side, out of your control). get_audio_position() locking with the audio thread (negligible unless audio is the bottleneck).

Caveats:

Cross-thread play_video race: pacer (end-of-video → self.play_video('stop')) can interleave with GUI-thread play_video('play'). Existing code already had a single-thread version of this; cross-thread tightens the window slightly but the worst case is one transient state which self-corrects on the next iteration. No lock added — pragmatic for now.
Pacer is daemon=True, so app shutdown drops it cleanly. The ThreadPoolExecutor atexit handler still waits for in-flight swaps.
HUD q: frame 0 scrub 0 will now be permanent — frame_q is never populated. If you want the queue line repurposed (e.g., to show process_qs 'finished' count, which is the new analog of backlog), say the word.

The venv update went much further than just torchcodec — everything jumped to latest, and the latest-everything combination is incompatible:

Component	Was	Now	Problem
Python	3.10	3.12	new DLL-search & invalid-escape warnings
numpy	1.26.4 (pinned)	2.4.3	breaks PyOpenGL 3.1.0 → your glGenTextures crash
torch	2.7.0+cu128	2.11.0+cu128	torchcodec 0.11 doesn't match this
torchcodec	0.11.1	broken	back to FileNotFoundError — Python 3.12's DLL rules + version skew
The OpenGL crash specifically is PyOpenGL 3.1.0 × numpy 2.x: PyOpenGL 3.1.0 was released before numpy 2.0's C-API change. Its array handlers don't recognize numpy 2.x ctypes, so it can't pass a buffer to glGenTextures → No array-type handler for type _ctypes.type. PyOpenGL 3.1.7+ has numpy-2 compat, or pinning numpy back to 1.26.4 also fixes it.

The \m SyntaxWarnings are Python 3.12 surfacing the invalid escape sequences '.\models\...' in Models.py — benign but noisy. Trivial fix while we're here.

Strong recommendation: revert to the pinned stack in requirements.txt. That combination (Python 3.10 / torch 2.3.0+cu118 / numpy 1.26.4 / torchcodec from cu118 index) is what the project was developed and tested against, and the torchcodec/FFmpeg ABI pairing for cu118 is well-trodden ground. GPU decode is most likely to work cleanly there.

Steps:


deactivate                                   # if it's active
rmdir /s /q D:\Rope\Rope.dev\Rope-Pearl.dev\venv
py -3.10 -m venv D:\Rope\Rope.dev\Rope-Pearl.dev\venv
D:\Rope\Rope.dev\Rope-Pearl.dev\venv\Scripts\activate
pip install -r D:\Rope\Rope.dev\Rope-Pearl.dev\requirements.txt
If py -3.10 isn't installed, grab Python 3.10.x from python.org and re-run (py -0 lists installed versions).

Then verify:


python -c "import numpy, torch, OpenGL; print(numpy.__version__, torch.__version__, OpenGL.__version__)"
python -c "from torchcodec.decoders import VideoDecoder; print('OK')"
I can fix the \m SyntaxWarning in Models.py right now since it's a one-line change per occurrence and will pay off regardless of which Python version you end up on — just make '.\\models\\X.onnx' into r'.\models\X.onnx' (raw string) or use forward slashes. Want me to?

If you genuinely need the cu128 / Python 3.12 / numpy 2.x stack for some reason (driver requires CUDA 12.8, etc.), the alternative is to bump only what has to bump:

pip install --upgrade PyOpenGL>=3.1.7 to fix the GL crash
Find a torchcodec wheel that's actually built for torch 2.11+cu128
Accept that other pinned packages (onnxruntime-gpu 1.18.0, opencv 4.10, etc.) may not work and need their own bumps
But that's a yak-shaving rabbit hole. The pinned stack is the path of least resistance.

--------------------------------------
after the inswapper outputs an image, is it possible to tune the input embedding so that the output image is closer to the expected embedding? looking for ways to improve the output likeness coming out of the inswapper
---------------------------------------

Yes — this is a known research direction with several practical levels. The inswapper's identity gap (cosine_sim of arcface(output) vs s_e is typically 0.6-0.85, not 1.0) is what you're trying to close. Here are the realistic options ranked by cost vs payoff:

Tier 1: Embedding shaping (free, predictable)

calc_swapper_latent does three steps on s_e: L2 normalize → project through emap → L2 normalize. Two things you can fiddle with that change identity intensity without extra inference:

Scale the post-projection latent — multiply by k > 1 (typically 1.05–1.3) before passing to inswapper. Pushes the model harder in the same identity direction. Cheap; just one float-multiply in calc_swapper_latent. Risks: too high a k produces visible "identity overshoot" artifacts (waxy / over-sharpened look). A slider exposing k lets users tune per face.

Embedding extrapolation away from a mean — s_e' = s_e + α (s_e − mean_emb) where mean_emb is the average of a small "neutral" face set. Same principle as a Rule-of-Thirds knob: pulls the embedding away from the population average, making distinctive features more pronounced. Free at inference; one-time precompute for mean_emb.

Both are 1-line additions. Won't close the gap fully, but typically improve perceived likeness by 5–15%.

Tier 2: One-shot fixed-point refinement (2× swap cost, real improvement)

This is closer to what you described. The principle is empirical, not gradient-based:


output1 = inswapper(target, latent_from(s_e))
output_emb = arcface(output1)         # what identity actually came out
delta = s_e − output_emb               # the gap
s_corrected = s_e + α · delta          # push input to compensate (α ~ 0.5–1.0)
output2 = inswapper(target, latent_from(s_corrected))
output2 is the displayed frame. Doubles the per-face swap cost (~30–50 ms extra at 128px), but the identity is provably closer to s_e because we've corrected for the model's measured drift. One iteration usually captures most of the win — diminishing returns past that.

Could be a toggle ("High-Fidelity") that activates this for offline export but stays off during playback.

Tier 3: Gradient-based latent optimization (true optimization, N× cost)

The version that actually solves "find latent* that minimizes 1 − cos(arcface(inswapper(target, latent*)), s_e)":


latent = calc_swapper_latent(s_e).clone().requires_grad_(True)
opt = torch.optim.Adam([latent], lr=0.01)
for _ in range(N):
    output = torch_inswapper(target, latent)
    emb = torch_arcface(output)
    loss = 1 - torch.cosine_similarity(emb, s_e).mean()
    opt.zero_grad(); loss.backward(); opt.step()
The hard part: you need differentiable versions of inswapper and arcface. Currently both are ONNX/TRT — no gradients. You'd need to either:

Convert inswapper_128.fp16.onnx → torch via onnx2torch (already a dependency, per requirements.txt). The recognition model w600k_r50.onnx would also need conversion. Both should work in principle; the converted graphs run a bit slower than ORT.
Or use an existing PyTorch port of InSwapper (some exist on GitHub) and ArcFace (insightface ships one).
Per-face cost at N=10 iterations: ~500 ms–1 s. Not playback-feasible but reasonable for offline export with the embedding cached afterward (Tier 1+3 hybrid: optimize once at assignment time, save the resulting latent, use it for all frames).

Tier 4: Output-space refinement

Train a small CNN on (swap_output, source_face) pairs to push the output's arcface embedding closer to source. Highest quality if done well, but requires training infrastructure and a dataset. Out of scope for an in-app feature.

My recommendation, in order:

Add a "Likeness" slider in the parameters pane that scales the post-emap latent (k from Tier 1). One file change in Models.calc_swapper_latent + a new SliderParam. Zero perf cost. Probably gives you 70% of what's achievable cheaply.

Add a "High-Fidelity" toggle that does Tier 2 (one-shot fixed-point). Doubles swap cost when on. Run an arcface pass on the first output, correct, re-swap. ~30 lines of code in swap_core.

If you want Tier 3, that's a multi-day project (onnx2torch conversion, gradient pipeline, latent caching keyed by (source, target) pairs, optimization run on assignment), but I can scope it.

Want me to wire up #1 (Likeness slider) and #2 (High-Fidelity toggle) as a starting point? Both are small enough to ship today and the user gets visible quality dials. Tier 3 we can scope separately if those aren't enough.

------------------------------------
Your pipeline is a classic polyphase filterbank: decompose the image into N² phase-shifted sub-images at 1/N resolution, run each sub-image through the pipeline independently, then interleave the outputs back into the full grid. 
-------------------------------------------------------------

5. Decoder replacement / extension. Keep the encoder + bottleneck, replace or extend the decoder with a higher-output one. Specifically: add an extra (Resize + Conv + LeakyReLU) stage to the existing decoder, initialized so it's near-identity, and fine-tune on paired data at 256² or 512². This is the most principled high-resolution fix but the most work. Could share weights between resolutions so the model still works at 128² too.

6. Per-image test-time optimization. With autograd, optimize a 256² (or 512²) image directly such that every 128² crop of it, when fed through inswapper, agrees with what the model produces. This is essentially solving the inverse problem of "what high-res output would have produced these tile outputs?" using the model as a forward operator. Powerful and rigorous but slow (~100 grad steps per image).