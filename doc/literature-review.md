# Comparision

## AFPILD: Acoustic footstep dataset collected using one microphone array and LiDAR sensor for person identification and localization

Year - 2024
Focus - Person identification and localization in the plane.

They claim, AFPILD data set has more samples than any other dataset. They considered shoe, clothing covariates in their dataset.

**Data**: Acoustic footstep audio using microphone array in middle. 

Mentioned that localization is less studied previously.

**Single footstep event**: Starts with single heel strike and till another leg's heel strike.

**Denoising**: multi-band spectral subtraction denoise algorithm

**Normalization**: Z-Score normalization

**Feature**: MFCC

**Model**: CRNN. SGD optimizer accompanied by the one-cycle learning rate scheduler

**Loss function**: Cross Entropy Loss function

Compared with SVM. Features for SVM are Spectracl Centroid, Spectracl Flatness and MFCC.

**Validation**: Timeshift the data

**Conclusion**: Shoe type has effect on recognition.


Insights: Change the train and test set and see the performance. Train with running and try for walking.

Futurework:
- Realtime noise
- Hands in pocket
- Floot material
- Transfer learning


Gap:
- Not mentioned how many times each action repeated.Repeatition is necessary to observe the normal gait.
- Not tested with overlapping frames.
- Only normal walking included. No other actions. fast walking is not there.
- Without shoe not tested.
- Only circular motion.




## Indoor Multiperson Detection and Recognition Through Footsteps: A Deep Learning Approach With Acoustic Signal Analysis

## Advanced acoustic footstep-based person identification dataset and method using multimodal feature fusion

Year - 2023
Person - 41
Single Person recognition


**pre-processing**:
- Multi band spectral subtraction denoising algorithm

**Augmentation**:
- Augment wrt a single footstep event

**Feature**:
- Raw waveforms, handcraft features, mel spectrogram. (Multiple features are fusioned in results)

**Model**: CNN model.

Results are discussed throughly, mentioning how different shoe type and wearings can affect the performance.



Year - 2024
Person - 18
Can recognize - 3 person

Trigger faking by ambient interference, a signal activity detection with a feature control
mechanism, which includes sound activity detection and noise sample filtering based on dynamic time warping (DTW).

Spectral subtraction is also used. Not clearly mentioned whats the purpose.

Note: Related work is good.

**Pre-Processing**: 
 - *Volume Activity Detection* works well for extracting active signal frame from noisy audio signal. 
 - Spectral Subtraction
 - DTW to differentiate featured data and noise and discard noisy frames.

Note: Employ VAD and spectral subtraction from this paper. Use it for overlapping window in your thesis.

**Feature**: Discrete Wavelet Transform

**Model and Prediction**: ResNet i.e a MulModel

Adam optimizer used.

SYM8 wavelet good performance.

Gap:




## Data collection methods
- **Wearable devices** : 
    - [EarGate: gait-based user identification with in-ear microphones](https://dl.acm.org/doi/10.1145/3447993.3483240)
- **Footstep audio microphone**:
    - [AFPILD](https://pdf.sciencedirectassets.com/272144/1-s2.0-S1566253523X00128/1-s2.0-S1566253523004979/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEFwaCXVzLWVhc3QtMSJGMEQCIH7xXskLbwcmYDtTwHq7PnorO2OWWciS3w9phLq0y06YAiAQj%2BUXzDKGk3BhFae%2Fv%2B2DqP%2Fzrl9CNfZalVL%2BPFC19CqzBQgVEAUaDDA1OTAwMzU0Njg2NSIMp4iJUfNpwoNMgIY3KpAFQ%2BOKKgl7TOQ%2BVwmHJ12zPNCDfKCsTcTUwqxp0FGtUtyuNu3CQX6Smsluf2tSumWb3K8traeg5%2BEDbPPhCqiENEU0YhENclDCOAxjmYmM8YqGNi9XlYo0t5Yl4dLLayyGQp1ExwO%2F22%2FP1IBa9Hga3TS%2Br1p1nCNei5MAmDmasfH0wriHLRE7zK0cyXzgyDYCq4DWmHIzTkw4pXdiF7d1xRf7Gz%2BzOJCpkG2rnHnFw7jfvMw79ZUaqFAb2caY1kImz2o7hVDnp7VOvNAjRvK9WTrviEtZrDDcnjboLlFllTvQlFDnjPjuZJGqvWcDR%2BMO41JEatJLG6ouNONS8aWHpJkmib%2FmJB5KZZcmzrT3Zm1HOtUAzL2Dg1j0t1wojikS5e0tJtHUhLLJ%2BhQqAh7jbwaCds0MfE2Yq%2BCayS8wLry8d6IgoOxk2kwPhxA1n6LR8qMM8nmJszC1wspZgUSoSBsfvUr8RQvV%2BZK3gVU0a6effGmA814by7zECfgqzaVWPEzbJDNrJPzYRPvzCrPiHZVmpFri%2BI7NRc2tFALHegaPNkWuINgR2fRAxNGTKSwiSAsPJNcCRmgfijsirj%2FI3EN4n1n%2F2VnggWT5E4Qkbqj%2BF6gbXTGUsZu9pA8EjSTeZ0lJtIv5BzNCQspRp96VJTV3GSPnxJ06npAkX%2FDztbjkz%2FR9neBEowg%2BGivQh%2BdcQXxNzHgeB41o%2FNN0uPssNkepA4b6H4euxqUyyxj5yDZZ7KvifvFEoBDN3W1XNR6VTA6y6MekXF%2F7uUkGtOAY7YVI7IQZ5p0u%2BIcwkOXov66ARYHCiwVbcNIjgzb9QGo0RsS1bertJOjgT3zhrcKrHr4ZDz6GfxUwkfrOLHjYCMMwkLr6swY6sgEC0hwzGUAzdhL2Q4j6D6BtSA7JC%2FABH0%2B14n0ADG%2FutB6PefnwmFcJuhCcQyoM8j77sluMjYyPfPHPeA4k3t3hwv3l1ijhM%2FobySwz2zyU%2FUu%2F3U3xydE%2BcsBC%2BIr1bgY5BNgv7t7mgCsDDeGez2snPi%2B1dMIs0Pxv8nneFdVCRHhb6fTJdonuBhGcrstV5ebRO0jjmZTwgaShC6e9TMAnbWWAjsjp3vW%2BeW051sLSqHQy&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240628T114241Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY7PLZ6PRP%2F20240628%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=c0c043bfba4039e467d9d4564c9778cf1df5e0579af2be001fbb4e39b028fd75&hash=1c08d8a55e7575db5c69db6855bf04c7b4b351ad9b02d1b1e1ba91a4b624e250&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S1566253523004979&tid=spdf-b770db24-fce4-4db1-9e8e-3b750e7c2446&sid=1228257f4d193141946942135db589c095a8gxrqb&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=020157560602580e5053&rr=89ad6a322bb95d74&cc=de)
    - [Indoor Multiperson Detection and Recognition Through Footsteps: A Dee](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10522611&tag=1)
- **Floor sensors**:
    - [Activity Recognition With Smart Polymer Floor Sensor: Application to ](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7452556)

- **Vibration sesnors**:
    - [Acoustic- and Radio-Frequency-Based Human Activity Recognition](https://www.mdpi.com/1424-8220/22/9/3125)


## Features used
- **MFCC**:
    - [AFPILD - 2024](https://pdf.sciencedirectassets.com/272144/1-s2.0-S1566253523X00128/1-s2.0-S1566253523004979/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEFwaCXVzLWVhc3QtMSJGMEQCIH7xXskLbwcmYDtTwHq7PnorO2OWWciS3w9phLq0y06YAiAQj%2BUXzDKGk3BhFae%2Fv%2B2DqP%2Fzrl9CNfZalVL%2BPFC19CqzBQgVEAUaDDA1OTAwMzU0Njg2NSIMp4iJUfNpwoNMgIY3KpAFQ%2BOKKgl7TOQ%2BVwmHJ12zPNCDfKCsTcTUwqxp0FGtUtyuNu3CQX6Smsluf2tSumWb3K8traeg5%2BEDbPPhCqiENEU0YhENclDCOAxjmYmM8YqGNi9XlYo0t5Yl4dLLayyGQp1ExwO%2F22%2FP1IBa9Hga3TS%2Br1p1nCNei5MAmDmasfH0wriHLRE7zK0cyXzgyDYCq4DWmHIzTkw4pXdiF7d1xRf7Gz%2BzOJCpkG2rnHnFw7jfvMw79ZUaqFAb2caY1kImz2o7hVDnp7VOvNAjRvK9WTrviEtZrDDcnjboLlFllTvQlFDnjPjuZJGqvWcDR%2BMO41JEatJLG6ouNONS8aWHpJkmib%2FmJB5KZZcmzrT3Zm1HOtUAzL2Dg1j0t1wojikS5e0tJtHUhLLJ%2BhQqAh7jbwaCds0MfE2Yq%2BCayS8wLry8d6IgoOxk2kwPhxA1n6LR8qMM8nmJszC1wspZgUSoSBsfvUr8RQvV%2BZK3gVU0a6effGmA814by7zECfgqzaVWPEzbJDNrJPzYRPvzCrPiHZVmpFri%2BI7NRc2tFALHegaPNkWuINgR2fRAxNGTKSwiSAsPJNcCRmgfijsirj%2FI3EN4n1n%2F2VnggWT5E4Qkbqj%2BF6gbXTGUsZu9pA8EjSTeZ0lJtIv5BzNCQspRp96VJTV3GSPnxJ06npAkX%2FDztbjkz%2FR9neBEowg%2BGivQh%2BdcQXxNzHgeB41o%2FNN0uPssNkepA4b6H4euxqUyyxj5yDZZ7KvifvFEoBDN3W1XNR6VTA6y6MekXF%2F7uUkGtOAY7YVI7IQZ5p0u%2BIcwkOXov66ARYHCiwVbcNIjgzb9QGo0RsS1bertJOjgT3zhrcKrHr4ZDz6GfxUwkfrOLHjYCMMwkLr6swY6sgEC0hwzGUAzdhL2Q4j6D6BtSA7JC%2FABH0%2B14n0ADG%2FutB6PefnwmFcJuhCcQyoM8j77sluMjYyPfPHPeA4k3t3hwv3l1ijhM%2FobySwz2zyU%2FUu%2F3U3xydE%2BcsBC%2BIr1bgY5BNgv7t7mgCsDDeGez2snPi%2B1dMIs0Pxv8nneFdVCRHhb6fTJdonuBhGcrstV5ebRO0jjmZTwgaShC6e9TMAnbWWAjsjp3vW%2BeW051sLSqHQy&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240628T114241Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY7PLZ6PRP%2F20240628%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=c0c043bfba4039e467d9d4564c9778cf1df5e0579af2be001fbb4e39b028fd75&hash=1c08d8a55e7575db5c69db6855bf04c7b4b351ad9b02d1b1e1ba91a4b624e250&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S1566253523004979&tid=spdf-b770db24-fce4-4db1-9e8e-3b750e7c2446&sid=1228257f4d193141946942135db589c095a8gxrqb&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=020157560602580e5053&rr=89ad6a322bb95d74&cc=de)
- **DWT**
    - [MultiPerson - 2024](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10522611)


## Model and prediction
- **CRNN**:
    - [AFPILD](https://pdf.sciencedirectassets.com/272144/1-s2.0-S1566253523X00128/1-s2.0-S1566253523004979/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEFwaCXVzLWVhc3QtMSJGMEQCIH7xXskLbwcmYDtTwHq7PnorO2OWWciS3w9phLq0y06YAiAQj%2BUXzDKGk3BhFae%2Fv%2B2DqP%2Fzrl9CNfZalVL%2BPFC19CqzBQgVEAUaDDA1OTAwMzU0Njg2NSIMp4iJUfNpwoNMgIY3KpAFQ%2BOKKgl7TOQ%2BVwmHJ12zPNCDfKCsTcTUwqxp0FGtUtyuNu3CQX6Smsluf2tSumWb3K8traeg5%2BEDbPPhCqiENEU0YhENclDCOAxjmYmM8YqGNi9XlYo0t5Yl4dLLayyGQp1ExwO%2F22%2FP1IBa9Hga3TS%2Br1p1nCNei5MAmDmasfH0wriHLRE7zK0cyXzgyDYCq4DWmHIzTkw4pXdiF7d1xRf7Gz%2BzOJCpkG2rnHnFw7jfvMw79ZUaqFAb2caY1kImz2o7hVDnp7VOvNAjRvK9WTrviEtZrDDcnjboLlFllTvQlFDnjPjuZJGqvWcDR%2BMO41JEatJLG6ouNONS8aWHpJkmib%2FmJB5KZZcmzrT3Zm1HOtUAzL2Dg1j0t1wojikS5e0tJtHUhLLJ%2BhQqAh7jbwaCds0MfE2Yq%2BCayS8wLry8d6IgoOxk2kwPhxA1n6LR8qMM8nmJszC1wspZgUSoSBsfvUr8RQvV%2BZK3gVU0a6effGmA814by7zECfgqzaVWPEzbJDNrJPzYRPvzCrPiHZVmpFri%2BI7NRc2tFALHegaPNkWuINgR2fRAxNGTKSwiSAsPJNcCRmgfijsirj%2FI3EN4n1n%2F2VnggWT5E4Qkbqj%2BF6gbXTGUsZu9pA8EjSTeZ0lJtIv5BzNCQspRp96VJTV3GSPnxJ06npAkX%2FDztbjkz%2FR9neBEowg%2BGivQh%2BdcQXxNzHgeB41o%2FNN0uPssNkepA4b6H4euxqUyyxj5yDZZ7KvifvFEoBDN3W1XNR6VTA6y6MekXF%2F7uUkGtOAY7YVI7IQZ5p0u%2BIcwkOXov66ARYHCiwVbcNIjgzb9QGo0RsS1bertJOjgT3zhrcKrHr4ZDz6GfxUwkfrOLHjYCMMwkLr6swY6sgEC0hwzGUAzdhL2Q4j6D6BtSA7JC%2FABH0%2B14n0ADG%2FutB6PefnwmFcJuhCcQyoM8j77sluMjYyPfPHPeA4k3t3hwv3l1ijhM%2FobySwz2zyU%2FUu%2F3U3xydE%2BcsBC%2BIr1bgY5BNgv7t7mgCsDDeGez2snPi%2B1dMIs0Pxv8nneFdVCRHhb6fTJdonuBhGcrstV5ebRO0jjmZTwgaShC6e9TMAnbWWAjsjp3vW%2BeW051sLSqHQy&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240628T114241Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY7PLZ6PRP%2F20240628%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=c0c043bfba4039e467d9d4564c9778cf1df5e0579af2be001fbb4e39b028fd75&hash=1c08d8a55e7575db5c69db6855bf04c7b4b351ad9b02d1b1e1ba91a4b624e250&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S1566253523004979&tid=spdf-b770db24-fce4-4db1-9e8e-3b750e7c2446&sid=1228257f4d193141946942135db589c095a8gxrqb&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=020157560602580e5053&rr=89ad6a322bb95d74&cc=de)

- **ResNet**
    - [MultiPerson 2024](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10522611)
- **Normal CNN** 
    - [AFPILD - 2023](https://pdf.sciencedirectassets.com/271505/1-s2.0-S0950705123X00033/1-s2.0-S0950705123000813/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEEIaCXVzLWVhc3QtMSJIMEYCIQDv7HIuQst1qLYLAsSQt3ZQKeXPZbjDN6G91LG0avbNgQIhAK9i4yfXSsio2pcutPRkxSSwm893ntVxhhVX4pCO4O6MKrsFCOv%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQBRoMMDU5MDAzNTQ2ODY1IgzP%2B%2FtSVWcQYEpX3SgqjwXOEVxhJ8B7nTEqyYMu0UWkPtty9dZfofanPTA%2F%2FuXVZNSGubcxH4oT6RVZQ3CApesdXYK0LHWZODMNhhH3eOydGIjFbNYuSzsxXqudxwKj9wjDOSWPWQ3P0EdGZqzn5U5jU%2B4T%2BaKNxGtQcC8tAwKARdsXtgph4%2FdgA%2F8slnhZrOtYjTBBare5jK9cayhHs%2FoQG67nygkr%2BUEQ1FPz13L7MGEWBLJ26mckWhBWG9QuHOjF%2FhxysSkVMrOeYZfyqpXMplXi5XCC33Av7MkCNV2q0eOwZ5SLoDLn1Q8YMK0NOXwi78BRIP53%2FBPtgBbPMvBOVRZhOdOQU3MsBK9tS3Fx9InMGKYOYRyl2jPO5Ipi%2FdUOSuwSlcwDqOogTdzNqUKuOCvfDh8k7gPHHpyJqc%2FFBNG0keOwblBq7ieR%2FvL5EBYh0O0LLC3BgsAb5eB8DktoNIFbHWd%2FRNfvCgdJzc4l8%2FBodxWabspb01IUQCRQFRhZJFcIDNiMaOOd383QT7lABapcKoVYBkQnZ8f79XdHTeF%2FP%2FPRPGQx4FMtx%2FywivOKBFIAQvCB0VIBNhg75IiwT%2BMdr2cqTW7m4d%2BxCPs98z4e8l4SO2M3LorvI1J%2FuGpgAf993KjjD3tEWAYpRIsLWq6qNtf6BTCjmwcMVZPPjS%2F3GSfeFl2jypyElyKt9g8zpsqUso4DZ%2BosYdeG2VTD3IV1iaeUq3WIfR%2FORfaUcj%2FQ0so4RwcYAr2c%2BOzRd%2B32bFZ5UZ%2FAXJnf9EAEzT3wmQ%2Bpo9CN4jHuGCs%2FKkNZ4t%2FG%2Fn%2F%2FqyaD3UOV9S%2BERjotFm2cPMzQQ%2BZR0AF2FUsy9f0a75exPBVJx845%2BtMA9M4IccX0%2F1uSlKCPxmWYMKrr9LMGOrAB07%2BJrswCSdgMjSB4d%2B333VaGhu76c5UjxGRirn0EPM9tAHUF4yrJdOHH2Ey32OHy%2FS3Wis%2BTQHpZz4GvC3UQ2T8fF8ZeVdSBEKIR2AEsJEsJz595V0vUlrZbd5iPOAE59rZghBVEP8%2BM61txlhHKFDXyzYBNh5Ey36%2Fn4A%2BHaCbsHvp2jF%2FKy1c3yUXSiYw%2BUj6FSQ2WMauWJvZrEUGD7Fet2DvSQXAOcmWx0dQCyfc%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240627T100927Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY5OUHUEFX%2F20240627%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=f1e77655dafaba0710c2609d7f07f4239ee56677a9bd653d33ded9ae3fbbfc91&hash=7fbb12d49b69999c0a88bfb3b344fb7916d7d16bafdfc00af3b058a056f3681a&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0950705123000813&tid=spdf-7d326003-17bb-4904-89b6-4ebe5aa08ae9&sid=0039cc845146e040ff69f0b49fc4cf4de88agxrqb&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=0201575606520f5b5750&rr=89a4a4419f46bbcc&cc=de)





### Research Questions
- What are the most effective features ?
- How does different actions such as running, fast walking impact the performance ?
- How does walking pattern, different footwears affect the performance ?
- How does the different sized frames affect the performance ?
- How does training sample size affect the performance ?
- How does noise impact the performance ?
