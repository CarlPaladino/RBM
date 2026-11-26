data is "emnist-digits-test.csv", and "emnist-digits-train.csv" from https://www.kaggle.com/datasets/crawford/emnist?select=emnist-digits-test.csv and https://www.kaggle.com/datasets/crawford/emnist?select=emnist-digits-train.csv

both csv's go into data/raw folder



Train new model:
adjust settings in CONFIG dictionary at top of main.py, then run:
python main.py

Run reconstruction:
python main.py --reconstruct-from "output/run\_cd20\_implementation\_label\_training/hidden\_512/checkpoints/epoch\_0200.npz" --samples 8 --output "output/reconstructions"

Run generation:
python main.py --generate-from "output/run\_cd20\_implementation\_label\_training/hidden\_512/checkpoints/epoch\_0200.npz" --class-name 7 --samples 16 --steps 5000 --output "output/digit7"

Run noisy reconstruction (original, clean reconstruction, noisy input, noisy reconstruction):
python main.py --reconstruct-from "output/run_mmnalefd/hidden_512/parameters.npz" --noise-level 0.1 --samples 8 --output "output/noisy_reconstructions"

Run generation comparisons:
python generate\_comparisons.py "output/run\_cd20\_implementation\_label\_training/hidden\_512/checkpoints/epoch\_0200.npz" --output "output/comparison"

--samples: amount of samples to reconstruct/generate
--class-name: class identifier to generate clamped from. can be ommitted from argument
--noise-level: is the probability of flipping each binary image pixel (0.1 means
10% on average). Labels are retained





