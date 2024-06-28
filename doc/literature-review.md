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
- Not tested with overlapping chunks ?
- Only normal walking included. No other actions. At least fast walking is not there.
- Without shoe not tested.
- Only circular motion.









## Data collection methods
- **In-Ear Microphones** : 
    - [EarGate: gait-based user identification with in-ear microphones](https://dl.acm.org/doi/10.1145/3447993.3483240)
- **Footstep audio with microphone**:
    - [AFPILD](https://pdf.sciencedirectassets.com/272144/1-s2.0-S1566253523X00128/1-s2.0-S1566253523004979/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEFwaCXVzLWVhc3QtMSJGMEQCIH7xXskLbwcmYDtTwHq7PnorO2OWWciS3w9phLq0y06YAiAQj%2BUXzDKGk3BhFae%2Fv%2B2DqP%2Fzrl9CNfZalVL%2BPFC19CqzBQgVEAUaDDA1OTAwMzU0Njg2NSIMp4iJUfNpwoNMgIY3KpAFQ%2BOKKgl7TOQ%2BVwmHJ12zPNCDfKCsTcTUwqxp0FGtUtyuNu3CQX6Smsluf2tSumWb3K8traeg5%2BEDbPPhCqiENEU0YhENclDCOAxjmYmM8YqGNi9XlYo0t5Yl4dLLayyGQp1ExwO%2F22%2FP1IBa9Hga3TS%2Br1p1nCNei5MAmDmasfH0wriHLRE7zK0cyXzgyDYCq4DWmHIzTkw4pXdiF7d1xRf7Gz%2BzOJCpkG2rnHnFw7jfvMw79ZUaqFAb2caY1kImz2o7hVDnp7VOvNAjRvK9WTrviEtZrDDcnjboLlFllTvQlFDnjPjuZJGqvWcDR%2BMO41JEatJLG6ouNONS8aWHpJkmib%2FmJB5KZZcmzrT3Zm1HOtUAzL2Dg1j0t1wojikS5e0tJtHUhLLJ%2BhQqAh7jbwaCds0MfE2Yq%2BCayS8wLry8d6IgoOxk2kwPhxA1n6LR8qMM8nmJszC1wspZgUSoSBsfvUr8RQvV%2BZK3gVU0a6effGmA814by7zECfgqzaVWPEzbJDNrJPzYRPvzCrPiHZVmpFri%2BI7NRc2tFALHegaPNkWuINgR2fRAxNGTKSwiSAsPJNcCRmgfijsirj%2FI3EN4n1n%2F2VnggWT5E4Qkbqj%2BF6gbXTGUsZu9pA8EjSTeZ0lJtIv5BzNCQspRp96VJTV3GSPnxJ06npAkX%2FDztbjkz%2FR9neBEowg%2BGivQh%2BdcQXxNzHgeB41o%2FNN0uPssNkepA4b6H4euxqUyyxj5yDZZ7KvifvFEoBDN3W1XNR6VTA6y6MekXF%2F7uUkGtOAY7YVI7IQZ5p0u%2BIcwkOXov66ARYHCiwVbcNIjgzb9QGo0RsS1bertJOjgT3zhrcKrHr4ZDz6GfxUwkfrOLHjYCMMwkLr6swY6sgEC0hwzGUAzdhL2Q4j6D6BtSA7JC%2FABH0%2B14n0ADG%2FutB6PefnwmFcJuhCcQyoM8j77sluMjYyPfPHPeA4k3t3hwv3l1ijhM%2FobySwz2zyU%2FUu%2F3U3xydE%2BcsBC%2BIr1bgY5BNgv7t7mgCsDDeGez2snPi%2B1dMIs0Pxv8nneFdVCRHhb6fTJdonuBhGcrstV5ebRO0jjmZTwgaShC6e9TMAnbWWAjsjp3vW%2BeW051sLSqHQy&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240628T114241Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY7PLZ6PRP%2F20240628%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=c0c043bfba4039e467d9d4564c9778cf1df5e0579af2be001fbb4e39b028fd75&hash=1c08d8a55e7575db5c69db6855bf04c7b4b351ad9b02d1b1e1ba91a4b624e250&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S1566253523004979&tid=spdf-b770db24-fce4-4db1-9e8e-3b750e7c2446&sid=1228257f4d193141946942135db589c095a8gxrqb&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=020157560602580e5053&rr=89ad6a322bb95d74&cc=de)
