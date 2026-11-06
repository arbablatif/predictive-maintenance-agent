# Learning notes

Plain-language explanations of the ideas and decisions in this project, so you can explain them to a client.

## The one-minute version

A machine's signal goes in and a maintenance ticket comes out. Signal processing and machine learning work out the numbers. A language model only chooses which tools to run and writes the summary. Code then checks that summary against the tool results. The main lesson is about honesty: the same model scores 100%, 97.5% or 87.2% depending only on how you test it.

## Bearing faults

A rolling bearing has an outer ring, an inner ring and balls between them. When a surface is damaged, every ball passing over the damage makes a tiny impact, and those impacts repeat at a steady rate.

That rate depends only on the bearing's shape and the shaft speed, so we can calculate it before measuring anything:

- **BPFO:** how often balls pass a fault on the outer ring (3.585 times per shaft turn here).
- **BPFI:** the same for the inner ring (5.415 times).
- **BSF:** how fast a ball spins on its own axis (2.357 times).

So each fault type has its own fingerprint frequency. Think of a stone stuck in a car tyre: the clicking speeds up as you drive faster, and its rate tells you it is in the tyre.

## Why the envelope spectrum is needed

You might expect a plain frequency analysis to show a peak at around 100 Hz. Usually it doesn't. Each impact is very short and makes the machine ring at a high natural frequency, perhaps 3,000 Hz, like tapping a glass. The 100 Hz part is how often the taps happen, not the pitch of the ring.

So we recover the tapping rate in three steps:

1. **Band-pass filter:** keep only the frequencies where the ringing is.
2. **Hilbert envelope:** trace the outline of the bursts, dropping the fast ringing inside them.
3. **Spectrum of that outline:** the tapping rate now appears as a clear peak.

Time-domain measures help too. **RMS** is overall vibration energy. **Kurtosis** and **crest factor** measure spikiness: damage makes spikes, so they rise even before you know which fault it is.

## Why the test split is the whole story

Each recording was cut into overlapping 1 s windows. Random splitting puts neighbouring, overlapping windows into both training and testing. That is like giving students the exam with half the answers already on their study sheet. It scores 100% and proves nothing.

Three levels of honesty:

| Test | What it asks | Recording accuracy |
|---|---|---|
| Random windows | Can it recognise a recording it has already seen? | 100% |
| Unseen load | Can it handle a new operating condition? | 97.5% |
| Unseen fault size | Can it handle a new physical defect? | 87.2% |

You asked for the load split, and it is a fair test of operating conditions. But CWRU reused the same damaged bearing at all four loads, so the model has met that exact defect. That is why I added the fault-size test. **If a client asks "how good is it?", give the 87.2% and explain why it's rough.** Depending on the filter band it ranges from 76.9% to 89.7%.

Another subtle leak: choosing the filter band using all the data lets test data influence a design decision. So the band is chosen inside each fold, from that fold's training data alone. Doing this properly changed the fault-size score from 79.5% to 87.2%, which shows how much that one setting matters.

## What checking the raw data found

Three problems in the CWRU files would have quietly damaged results:

- **Wrong sample rate.** Healthy files are recorded four times faster than the fault files. To prove it, I watched the shaft peak as speed changed. At the right rate it followed the speed. At the wrong rate it stayed fixed at 30 Hz, because that peak was really the 120 Hz electrical hum.
- **A hidden duplicate.** `99.mat` contains an exact copy of the recording in `98.mat`, which would have leaked data between loads.
- **Missing speeds.** Two files have no RPM value, so a documented nominal value is used.

Client takeaway: data checks often matter more than model choice.

The sample rate trap came back in the app. The upload form defaulted to 12,000 Hz, and that quietly replaced the known 48 kHz rate of the healthy files. A healthy recording then came out as low confidence, needing engineer review. Now the form starts empty, and a rate that conflicts with a known recording is refused rather than used. A default value in a form is a guess too.

## Why gradient boosting

It builds many small decision trees, each correcting the errors of the ones before. It works well on a few dozen hand-made features and small datasets, trains in seconds on a laptop, and needs no GPU. A neural network would have little to learn from with only 40 recordings.

## Remaining useful life

The goal is to predict how many cycles (flights, here) an engine has left.

- **The 125 cycle cap.** Early in life, an engine shows no sign of wear, so nothing in the data separates 300 cycles left from 200. We therefore label anything above 125 as 125. The model learns "healthy" and then a countdown.
- **Trailing windows.** Each prediction uses only the last 30 cycles of each sensor: level, spread and trend. It never sees the future, and a test confirms this.
- **No age feature.** A model that knows the age can score decently by learning the fleet's average life without reading the engine's condition.
- **The baseline.** "Average life of engines this old, minus current age" uses no sensors. Beating it (RMSE 33 down to 13) shows the sensors add real value.
- **The NASA score.** It punishes late predictions more than early ones, because predicting too much life left means a failure in service. It grows exponentially, so one bad miss can dominate. Lower is better.
- **Prediction intervals.** Instead of "6 cycles", we say "4 to 7, 80% of the time". On test data the interval held for 79% of engines, so it is well calibrated.

**The limit to be upfront about.** The official test RMSE of 13.27 looks excellent, but NASA's test set has few healthy engines. With engines cut at random ages, the error for those with more than 145 cycles left is 70. The model is strong near failure, and further out it can only say "healthy". That's fine for maintenance, but it should never be sold as a long-range forecast.

## Why the agent can't estimate a bearing's life

The two datasets describe different machines, and the bearing data never runs to failure. A bearing RUL would be a number with nothing behind it. So `estimate_rul` refuses vibration data, the ticket says the figure is not available, and the text review rejects a life figure the model writes anyway. The evaluation includes a trap case asking for one.

Saying "we can't know that" is a selling point. It's what makes the numbers it does give believable.

## How the agent is kept honest

Telling a model "don't invent numbers" is a request, not a guarantee. So there are four layers:

1. **Keep important fields out of its hands.** Urgency, confidence and deadline are filled in by code. Even if the model writes "critical", the ticket shows the policy value.
2. **Check what it does write.** Every number in its text must match a tool result, allowing for rounding and percentages. Spelled-out numbers count too.
3. **Be stricter on deadlines.** "3 days" fails even if a 3 is somewhere in the results. So do "next month", "a few days" and "by Friday", which have no number to check.
4. **Check the meaning in simple ways.** A bearing ticket can only say its remaining life is not available. A healthy result can't be written up as a fault, and a fault can't be denied.

A failed draft gets one retry. If it still fails, the ticket stays visible, marked UNVERIFIED, because hiding a failure is worse than showing it.

**What this does not prove.** It proves a number came from a tool, not that the number is right. Layers 2 to 4 are pattern matching, so unusual wording can still get past them. That's why layer 1 matters most: the fields a technician acts on never depend on the model's wording.

**A story worth telling.** The first live run needed 4 retries, which looked like the model was making mistakes. Reading the drafts showed the opposite: the model was right, and my checker wrongly rejected `outer_race` because it expected `outer race`. Never trust a summary table without reading what's behind it.

**A second story, against myself.** The check first scored 40 of 40 on invented claims, and all 121 tests passed. A later code review still found "within thirteen months", "by Friday" and "Remaining useful life is 7 days" getting through, and a healthy bearing could be written up as a severe fault. I had written the tests and the invented claims myself, so they only covered what I'd already thought of. If a client asks what a perfect score means, the honest answer is that it only covers what was tried. Having someone try to break the check taught more than the score did.

## Confidence

Model probability alone is not confidence. On the false alarm the model was 99.5% sure and wrong. So the label is a fixed rule: "high" needs a confident classifier *and* a visible peak at the matching fault frequency. Two independent kinds of evidence must agree.

Across both hard tests, all 38 high-confidence calls were right. The false alarm got "medium", which leads to a second measurement rather than a needless bearing change. Ball faults can't reach "high" on this rig, because their envelope peaks are too weak to confirm.

## Engineering choices

- **Mock mode:** the whole system runs with no API key, so it can be demonstrated and tested for free. It uses simple rules in place of the model, so it proves the plumbing, not the model's judgement.
- **Disk cache:** a repeated request costs nothing. It also let me re-score the live run after a bug fix with no new spending.
- **Budget limit:** `--max-calls` stops a live run in code.
- **Synthetic samples:** the repo contains no dataset files, so generated signals make the demo work straight after cloning. They are always labelled synthetic.
- **Policy table:** urgency values are in one small file because they are business decisions, not machine learning. They are placeholders for a client's reliability engineer to set.

## Questions a client might ask

**Will this work on my machines?** Not as it stands. The pipeline transfers, but the model would need retraining on your machines, especially on plenty of healthy data. The fault frequency calculation carries over to any bearing with known geometry.

**How much data would you need?** Far more healthy data than this project had, across normal operating conditions, plus confirmed fault cases from maintenance records. Healthy data is the cheapest place to start.

**Why not let the AI do the analysis?** Language models are unreliable at calculation and can't be audited. Here each number traces back to a known calculation.

**Can it tell me when a bearing will fail?** No. That needs run-to-failure history, which this bearing dataset doesn't have. It can show fault type, and repeated measurements over time could build that history.

**What happens when it doesn't know?** It lowers confidence, warns about unusual inputs and recommends a human review. The weak spot is that it can't say "this is a fault type I've never seen".
