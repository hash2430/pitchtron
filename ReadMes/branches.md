Pitchtron
============
![](pitchtron_logo.png)

* Prosody transfer toolkit where you don't need stylish training DB.
* We can transfer Korean dialects(Kyongsang, Cheolla) and emotive prosodies.
* Hard pitchtron is for strictly transferring the prosody thus, the sentence structure of reference audio and target sentence better match.
* Soft pitchtron pursues natural sounding prosody transfer even the reference audio and target sentence are totally different in content.  

Differences of three branches
====================
* All three branches provided here are for **prosody transfer.**
* You can generate speech of desired style,sentence and voice.
    * The speaker of reference audio can be anyone and that person is not necessary to be included in the training data.
    * The target speaker (the voice of synthesized audio) must be included in the training data.
* Using hard and soft pitchtorn, you can synthesize in 'Kyongsang' dialect, 'Cheolla' dialect and emotional style even if the model is only trained with plain, neutral speech.
* On the other hand, for global style token, you need the DB of desired style during training time.
* I proposed this **pitchtron** in order to speak in Korean Kyongsang anc Cheolla dialect.
* The DB of these dialects are very limited and 'pitch contour' is key to referencing them naturally. This is also true of many other pitch-accented language(Japanese), tonal langauge(Chinese) and emotional speaking style.


|                | Temporal resolution | Linear control | Vocal range adjustment | Non-parallel referencing | Unseen style support | Dimension analysis requirement |
|----------------|---------------------|----------------|------------------------|--------------------------|----------------------|--------------------------------|
| GST            | X                   | X              | X                      | O                        | X                    | O                              |
| Soft pitchtron | O                   | *              | O                      | O                        | O                    | X                              |
| Hard pitchtron | O                   | O              | O                      | **                       | O                    | X                              |
* *: Soft pitchtron will let you control the pitch as long as it can sound natural. If it is out of vocal range of target speaker, it will be clipped to make natural sound.
* **: Hard pitchtron allows limited non-parallel referencing.
    * Limited non-parallel: the text can differ, but the structure of the sentence must match.
   
|           | Sentence                                        |
|-----------|-------------------------------------------------|
| Reference | "아니요 지는 그짝허고 이야기허고 싶지 않아요"   |
| Target    | "그래요 갸는 친구허고 나들이가고 싶은것 같아요" |
* Meaning of each column
1. Temporal resolution: Can we control the style differently by timestep?
2. Linear control: Can I control exactly to what amount the pitch(note) is going to be scaled? I don't have to explore on the embedding space to figure out the scale change in embedding dimension as the input changes?
3. Vocal range adjustment: If the vocal range of reference speaker and target speaker are drastically different, can I reference naturally in target speaker's vocal range?
4. Non-parallel referencing: If the reference sentence and target sentence are different, can I synthesize it naturally?
5. Unseen style support: If the desired reference audio is of the style that has never been seen during training, can it be transferred naturally?
6. Dimension analysis requirement: Do I have to analyze which token/dimension controls which attribute to have control over this model?



**1. Soft pitchtron**
---------------------
* This branch provides unsupervised prosody transfer of parallel, limited non-parallel and non-parallel sentences.
* Parallel: Reference audio sentence and target synthesis sentence matches.
* Limited non-parallel: mentioned above.  
* Non-parallel: Reference audio sentence and target synthesis sentence need not match.
* Similar to Global style token, but there are several advantages.
    * It is much more robust to styles that are unseen during training.
    * It is much easier to control.
        * You don't have to analyze tokens or dimensions to see what each token does.
        * You can scale the pitch range of reference audio to fit that of target speaker so that inter-gender transfer is more natural.
        * You can also control pitch for every phoneme input
* Pitch range of reference audio is scaled to fit that of target speaker so that inter-gender transfer is more natural.
* Your control over pitch is not so strict that it will only scale to the amount it sounds natural.

![Soft pitchtron](soft_pitchtron.png)

**2. Hard pitchtron**
-------------------------
* This branch provides unsupervised parallel and 'limited non-parallel' unsupervised prosody transfer.
* Instead, the rhythm and pitch are exactly the same as reference audio.
* Pitch range of reference audio is scaled to fit that of target speaker so that inter-gender transfer is more natural.
* You have strict control over pitch range, to the amount where it will scale even if it results in unnatural sound.

![Hard pitchtron](Hard_pitchtron.png)

**3. Global style token**
---------------------------
* Global style token implementation. 
[Global style token](https://arxiv.org/abs/1803.09017)
* Unlike pitchtron, global style token tend to work well only for the styles that are seen during training phase.
* Pitch range cannot be scaled, resulting noisy sound if reference audio is out of vocal range of target speaker.
* Since it is not robust to new style unseen during training, it sometimes generates speech with too loud energy or too long pause.

![GST](gst.png)
