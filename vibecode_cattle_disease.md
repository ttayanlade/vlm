# Vibecoding a cattle photo model

Vibecoding means you describe what you want in ordinary English, and an AI helper writes the code.

You will teach a computer to look at a photo of cattle and place it in one of three groups:

1. **Healthy**
2. **Lumpy skin**
3. **Foot and mouth**

You will train two models on the same photos, then compare them.

1. A **plain model**. It starts with no knowledge of photos and learns only from these cattle pictures.
2. **ResNet50**. It already knows how to look at ordinary photos. You teach it these three groups.

The photos come from a public collection, the [Cattle diseases dataset](https://www.kaggle.com/datasets/devang03mgr/cattle-diseases-datasets). Register a Kaggle account so you can download the zip.

You will not write the program yourself. This page has the instructions. Each step also has a prompt in a gray box. Copy that prompt, then either click the sparkle in Colab or paste it into the LLM of your choice. The helper writes the Python. You paste that Python into Colab and run it.

> **This is a classroom exercise.** The program can be wrong. It must not be used to decide whether a real animal is sick. A veterinarian makes that decision.

## Start with an empty Colab

1. Open [Google Colab](https://colab.research.google.com) and create a **new notebook**.
2. Turn on a GPU before you run anything: **Runtime → Change runtime type → T4 GPU → Save**.
3. On Kaggle, download the cattle diseases zip. In Colab, open the folder icon on the left and upload that zip. The file should be named `archive.zip` and sit at `/content/archive.zip`.
4. Work down this page one step at a time. For each step:
   - Read the section called **For you**.
   - Copy the gray prompt. On GitHub, use the copy icon at the top right of the box.
   - In Colab, click **+ Code** to add a new empty cell.
   - Do not paste the prompt in as the program, and do not press play on the English. Get the Python in either of these ways:
     - **Click the sparkle** on the empty Colab cell, paste the prompt, and accept the Python it writes into that cell.
     - **Or paste the prompt into the LLM of your choice** (ChatGPT, Claude, Gemini, Cursor, or another chat). Copy only the Python it returns, and paste that Python into the Colab cell.
   - Press the play button. Read what it prints. Compare it with **You will know it worked when**.
5. Each step needs the one before it. If the page restarts, run the finished cells again from Step 1.

If the first line in a code cell is an ordinary sentence, delete the sentences and keep the Python.

| Word | Meaning |
| --- | --- |
| Prompt | The note you copy. Click the Colab sparkle, or paste it into the LLM of your choice |
| Cell | One box in Colab. The play button runs a code cell |
| Group | One category of photo. There are three |
| Model | The program that learns from the photos |
| Train | Photos the model learns from. About 70 percent |
| Validation | Photos used during teaching to see if the model is improving, and to decide when to stop. About 15 percent |
| Test | Photos kept aside until the end. The final score uses only these. About 15 percent |
| Epoch | One pass through the training photos |
| Plain model | The first model. It learns only from these cattle pictures |
| ResNet50 | The second model. It already knows how to look at ordinary photos |

The zip is about 270 megabytes. Each teaching step often takes 10 to 25 minutes. Leave the tab open while a cell runs.

## Step 1. Download the pictures

### For you

The zip contains a folder named `Cows datasets`. Inside it are three folders: `foot-and-mouth`, `healthy`, and `lumpy`. The folder name is the group.

The prompt already looks for `/content/archive.zip`. Change that line only if your file is somewhere else, or if you have a direct download link.

### Copy this prompt

```text
Write one notebook cell. Python only. No explanation, and no lines that start with ! or %.
If you use a function, declare DATA_DIR global before you assign it.
If something a beginner can fix goes wrong, print one sentence and return.

The zip path or link is:
/content/archive.zip

If that is a local zip path and the file exists, copy it to cattle_diseases.zip. Do not download.
If it is a Google Drive link, install gdown if needed and download with gdown.download(..., fuzzy=True).
If it is a Dropbox link containing dl=0, change dl=0 to dl=1, then download with urllib.
Otherwise download with urllib and print progress about every 10 percent.
If the path is missing, print: The zip was not found. Upload archive.zip to Colab so it is at /content/archive.zip, or paste a direct download link, then run this cell again.
If a download fails, print: The download failed. Check that you are online and that the link points straight at the zip file. Then run this step again.
If the file does not start with PK, print: That link did not download a zip file. It may be a webpage. Ask your teacher for a direct download link.
If it is not a valid zip, print: The file downloaded, but it is not a valid zip. Ask your teacher for a new link.

Unpack into cattle_data. Find the folder that contains foot-and-mouth, healthy, and lumpy. It is named Cows datasets, with a space. Save that path as DATA_DIR.
If those folders are missing, print: The zip unpacked, but the three folders foot-and-mouth, healthy, and lumpy were not found. Tell your teacher.

Count .jpg, .jpeg, and .png files. Print DATA_DIR, then one sentence each for Foot and mouth, Healthy, and Lumpy skin, then the total. Use thousands separators, like 1,291.
Do not train or rename the folders.
```

### You will know it worked when

You see these counts:

- Foot and mouth: 746 pictures
- Healthy: 1,291 pictures
- Lumpy skin: 1,207 pictures

The total is 3,244 pictures. If the cell says the zip was not found, upload `archive.zip` in Colab and run the cell again.

## Step 2. Look through the pictures

### For you

Look at the counts, the sizes, and a few real photos before any training.

Some photos are a whole animal in a field. Some are a close-up of a nose or a mouth. Lighting and backgrounds change. Some file names mention a different disease than the folder. Trust the folder.

While you look, notice which groups are close-ups, whether the backgrounds match, and which group has the fewest pictures.

### Copy this prompt

```text
Write one notebook cell. Python only. No explanation, and no lines that start with ! or %.
Do not download or train.
Use DATA_DIR from the previous cell. If it is missing, print "Go back to Step 1 and finish the download first." and return.
Set global class_names to ["foot-and-mouth", "healthy", "lumpy"].

Count .jpg, .jpeg, and .png files. Print short sentences with thousands separators, like 1,291:
- how many groups and pictures in total
- the count for Foot and mouth, Healthy, and Lumpy skin
- which group is largest and which is smallest
- the width and height of two pictures from each group
- that the next step makes every picture the same size
- that the folder name is the answer, so ignore messy file names
- to look at the examples and notice zoom, background, and lighting

Use Pillow and matplotlib. Open photos in color and close the files.
Show a bar chart titled "How many pictures are in each group", with display names and the count above each bar.
Show 3 photos from each group, titled "Three example pictures from each group", with the axes hidden.
Call plt.show() for each figure.
```

### You will know it worked when

You see the same three counts, a bar chart, and three example photos from each group. The writing on the screen is ordinary English.

## Step 3. Prepare the train, validation, and test pictures

### For you

Every photo has to be the same size. The helper then splits them into three piles:

- **Train**, about 70 percent. The model learns from these.
- **Validation**, about 15 percent. During teaching, the model is scored on these so it can stop when it stops improving.
- **Test**, about 15 percent. Training does not use these. Both models are scored on this same pile, so the comparison is fair.

The split uses seed 42, so the class shares the same three piles. Each disease appears in all three. Training pictures also get a small flip or tilt. The animal is still the same animal. Validation pictures and test pictures stay as they are.

### Copy this prompt

```text
Write one notebook cell. Python only. No explanation, and no lines that start with ! or %.
Do not download or train. Install tensorflow only if importing it fails.
If DATA_DIR is missing, print "Go back to Step 1 and finish the download first." and return.

Leave these global: IMG_SIZE=224, BATCH_SIZE=32, SEED=42, class_names, train_ds, val_ds, test_ds, data_augmentation.
Set the TensorFlow and Keras seed to 42.
Set class_names to ["foot-and-mouth", "healthy", "lumpy"].

Split the photos into train, validation, and test. Do not use validation_split, because that only makes two piles.
For each class folder, list the .jpg, .jpeg, and .png files and shuffle that list with numpy default_rng(42).
Put the first 70 percent in train, the next 15 percent in validation, and the last 15 percent in test.
If rounding leaves a picture out, put it in train. A picture belongs to only one pile.

Build train_ds, val_ds, and test_ds from those file lists.
Read each file with tf.io.decode_image so both jpg and png work, resize to 224 by 224, keep color, and use integer labels in class_names order.
Batch size is 32.
Shuffle train_ds. Do not shuffle val_ds or test_ds.
Prefetch train_ds. Cache, then prefetch, val_ds and test_ds. Do not cache train_ds. Do not repeat any dataset.

data_augmentation is a Keras Sequential named simple_photo_changes, in this order: RandomFlip horizontal, RandomRotation 0.05, RandomZoom 0.1, RandomContrast 0.1.
Do not put those changes inside the datasets. The teaching step places data_augmentation inside the model, so the changes happen only during training.

Print the groups as Foot and mouth, Healthy, and Lumpy skin.
Print how many training batches, validation batches, and test batches there are, and that each batch holds up to 32 pictures.
Print that about 70 percent are for training, 15 percent are for validation, and 15 percent are for the test.
Print that every picture is 224 by 224 and kept in color.
Print that flips and slight tilts happen only to training pictures. Validation and test pictures stay as they are.

Show 5 versions of one training picture after those changes. Clip pixels to 0 to 255, draw them in color, title the figure "The same training picture with small random changes", and call plt.show().
```

### You will know it worked when

You see the three group names, a count of training batches, validation batches, and test batches, and one training photo repeated five times with small differences.

## Step 4. Train the plain model

### For you

This model is a small photo model built for this lesson. It starts with no knowledge of photos. It learns only from the cattle pictures.

It trains on the training pictures and is scored on the validation pictures. It does not look at the test pictures in this step.

Groups with fewer pictures get a stronger voice, so the largest group does not take over the lesson. The ResNet step later uses these same weights and the same 5 epochs, so the comparison stays fair.

Leave the cell running. A moving progress bar means it is working. This often takes 10 to 25 minutes on a GPU. The cell may finish before 5 epochs, when the validation score stops improving. It keeps the best version.

Write down the test percent from the next step. You will compare it with ResNet.

### Copy this prompt

```text
Write one notebook cell. Python only. No explanation, and no lines that start with ! or %.
Do not rebuild train_ds, val_ds, or test_ds. Do not use test_ds in this cell. Do not build ResNet.
These must already exist: DATA_DIR, train_ds, val_ds, test_ds, class_names, data_augmentation, IMG_SIZE, BATCH_SIZE, SEED. If one is missing, print "Go back and run the prepare step first. This step needs the train, validation, and test pictures." and return.
Leave simple_model, simple_history, and class_weight global.

Count .jpg, .jpeg, and .png files in each class folder, in class_names order.
class_weight for class i = total pictures / (number of classes * pictures in class i).
Print each group as Foot and mouth, Healthy, or Lumpy skin, with its count (thousands separators) and weight to two decimals.
Then print: Smaller groups get a higher weight so the model does not ignore them.

Build one plain model. It must not be pretrained, and it must not be ResNet.
Input (IMG_SIZE, IMG_SIZE, 3), then the existing data_augmentation, then a Rescaling layer that divides by 255.
Then Conv2D 32 filters, kernel 3, relu, padding same, then MaxPooling2D.
Then Conv2D 64 filters, kernel 3, relu, padding same, then MaxPooling2D.
Then Conv2D 128 filters, kernel 3, relu, padding same, then GlobalAveragePooling2D.
Then Dense 128 relu, Dropout 0.3, and a softmax Dense with one unit per class.
Name this model simple_model.

Compile with Adam at learning rate 0.001, sparse_categorical_crossentropy, and accuracy.
Before fit, print: Training the plain model has started. This often takes 10 to 25 minutes. A moving progress bar means it is working.
Fit train_ds with validation_data=val_ds, epochs=5, class_weight, shuffle=False, and verbose=1. Store the fit result in simple_history.
Use EarlyStopping on val_accuracy, patience 2, restore_best_weights=True.
Save the model as simple_cattle_model.keras.
Print the best validation accuracy as a percent.
Print: The plain model was saved as simple_cattle_model.keras.
Print: This model is for learning. It is not a veterinarian.
Do not use PyTorch. Do not evaluate the test set here.
```

### You will know it worked when

The cell prints a validation percent and says the model was saved as `simple_cattle_model.keras`.

A validation score above 33 percent is better than guessing. The next step is the test score.

## Step 5. See how the plain model did

### For you

Training used the training pictures. The validation pictures decided when to stop. This step scores the plain model on the **test** pictures, which it has not used yet.

You will see the training and validation scores over time, a grid of which groups get mixed up, and individual test photos with the guess next to the real group.

Numbers on the diagonal of that grid are right. Numbers off the diagonal are mix-ups. Look back at the Step 2 photos for a reason. Keep this test percent. Step 8 compares it with ResNet.

### Copy this prompt

```text
Write one notebook cell. Python only. No explanation, and no lines that start with ! or %.
Do not train, download, or rebuild the datasets. Do not build ResNet.
Needs simple_model, simple_history, test_ds, and class_names. If one is missing, print "Train the plain model first, then come back to this step." and return.
Leave simple_test_accuracy global. It must be the test percent from 0 to 100.
Display names: foot-and-mouth as Foot and mouth, healthy as Healthy, lumpy as Lumpy skin.

The history chart uses simple_history and is training versus validation. The photos, the percent correct, and the mix-ups all come from the test set and simple_model. Do not use val_ds for those.
The plain model already rescales pixels inside the model. Do not divide by 255 again, and do not use ResNet preprocessing. Collected images are still 0 to 255 color pictures. Draw them directly.

One matplotlib figure, then plt.show():
- Left, title "Plain model: how often it was correct": training and validation accuracy as percent correct from 0 to 100. Lines: "Train" and "Validation". X-axis "Epoch". Y-axis "Percent correct".
- Right, title "Plain model: how far off it was": training and validation loss, same line names and x-axis. Y-axis "Amount of error".

Loop through test_ds once, keep the images and true labels together, predict with simple_model, and take argmax.
Set simple_test_accuracy to the real percent correct on these test pictures.
Print: Plain model: on the test pictures, it was correct about N percent of the time.
Print: Guessing one of the three groups at random would be right about 33 percent of the time.
Print: These test pictures were not used for training, and they were not used to decide when to stop.

Use scikit-learn's confusion_matrix (install scikit-learn if needed). Draw it with matplotlib, not seaborn. Title "Plain model: where it got confused on the test pictures". X-axis "What the model guessed". Y-axis "The real group". Display names and the count in each square. plt.show().
Print the most common mix-up: real group, guessed group, and how many pictures. If there is none, say the plain model did not mix up any groups.
Do not print a classification report.
For each group, print how many of its test pictures the plain model found, out of how many, and the percent.

Show 12 test pictures, or fewer if needed, chosen with numpy default_rng(42), in a 3 by 4 grid. Title each with the guess and the real group. Green text when they match, red when they do not. Figure title: Plain model. Green means the guess matches. Red means it does not. Clip to 0 to 255. plt.show().
Then print:
Green titles are correct guesses. Red titles are mistakes.
This is a classroom model. It must not be used to diagnose a real animal.
```

### You will know it worked when

- You see charts titled "Plain model", with lines named Train and Validation.
- You see a number grid about where the plain model got confused.
- You see 12 test photos. Green titles are right. Red titles are mistakes.
- A sentence gives the plain model's test percent.

## Step 6. Train ResNet

### For you

ResNet50 already knows how to look at ordinary photos. This step keeps that skill and adds a small ending that sorts each photo into the three groups. The first run downloads that starter model.

Use the same training pictures, the same validation pictures, the same class weights, and the same 5 epochs as the plain model. Do not look at the test pictures while training.

Leave the cell running. A moving progress bar means it is working. This often takes 10 to 25 minutes on a GPU. The cell may finish before 5 epochs, when the validation score stops improving. It keeps the best version.

### Copy this prompt

```text
Write one notebook cell. Python only. No explanation, and no lines that start with ! or %.
Do not rebuild train_ds, val_ds, or test_ds. Do not use test_ds in this cell. Do not replace simple_model or simple_history.
These must already exist: DATA_DIR, train_ds, val_ds, test_ds, class_names, IMG_SIZE, BATCH_SIZE, SEED, class_weight. If one is missing, print "Train the plain model first. ResNet should use the same class weights and the same pictures." and return.
Leave resnet_model and resnet_history global.

Build a new augmentation Sequential. Do not reuse the data_augmentation layer, because it already belongs to the plain model. Use the same four layers: RandomFlip horizontal, RandomRotation 0.05, RandomZoom 0.1, RandomContrast 0.1.

Build one ResNet model:
Input (IMG_SIZE, IMG_SIZE, 3), then that new augmentation, then tf.keras.applications.resnet.preprocess_input, then a frozen ResNet50 (include_top=False, weights="imagenet", pooling="avg", called with training=False), then Dense 128 relu, Dropout 0.3, and a softmax Dense with one unit per class.
Pixels stay 0 to 255. Do not also divide by 255. Use ResNet preprocessing.
Name this model resnet_model.

Compile with Adam at learning rate 0.001, sparse_categorical_crossentropy, and accuracy.
Before fit, print: Training ResNet has started. This often takes 10 to 25 minutes. A moving progress bar means it is working.
Fit train_ds with validation_data=val_ds, epochs=5, class_weight, shuffle=False, and verbose=1. Store the fit result in resnet_history.
Use EarlyStopping on val_accuracy, patience 2, restore_best_weights=True.
Save the model as resnet_cattle_model.keras.
Print the best validation accuracy as a percent.
Print: ResNet was saved as resnet_cattle_model.keras.
Print: This model is for learning. It is not a veterinarian.
Do not unfreeze ResNet50. Do not use PyTorch. Do not evaluate the test set here.
```

### You will know it worked when

The cell prints a validation percent and says the model was saved as `resnet_cattle_model.keras`.

The next step scores ResNet on the same test pictures the plain model already saw.

## Step 7. See how ResNet did

### For you

This is the same kind of score as Step 5, on the same test pictures, for ResNet. Keep this test percent. The next step puts it next to the plain model.

### Copy this prompt

```text
Write one notebook cell. Python only. No explanation, and no lines that start with ! or %.
Do not train, download, or rebuild the datasets. Do not change simple_model or simple_test_accuracy.
Needs resnet_model, resnet_history, test_ds, and class_names. If one is missing, print "Train ResNet first, then come back to this step." and return.
Leave resnet_test_accuracy global. It must be the test percent from 0 to 100.
Display names: foot-and-mouth as Foot and mouth, healthy as Healthy, lumpy as Lumpy skin.

The history chart uses resnet_history and is training versus validation. The photos, the percent correct, and the mix-ups all come from the test set and resnet_model. Do not use val_ds for those.
ResNet already includes its preprocessing. Do not divide by 255 or preprocess again. Collected images are still 0 to 255 color pictures. Draw them directly.

One matplotlib figure, then plt.show():
- Left, title "ResNet: how often it was correct": training and validation accuracy as percent correct from 0 to 100. Lines: "Train" and "Validation". X-axis "Epoch". Y-axis "Percent correct".
- Right, title "ResNet: how far off it was": training and validation loss, same line names and x-axis. Y-axis "Amount of error".

Loop through test_ds once, keep the images and true labels together, predict with resnet_model, and take argmax.
Set resnet_test_accuracy to the real percent correct on these test pictures.
Print: ResNet: on the test pictures, it was correct about N percent of the time.
Print: Guessing one of the three groups at random would be right about 33 percent of the time.
Print: These are the same test pictures used to score the plain model.

Use scikit-learn's confusion_matrix (install scikit-learn if needed). Draw it with matplotlib, not seaborn. Title "ResNet: where it got confused on the test pictures". X-axis "What the model guessed". Y-axis "The real group". Display names and the count in each square. plt.show().
Print the most common mix-up: real group, guessed group, and how many pictures. If there is none, say ResNet did not mix up any groups.
Do not print a classification report.
For each group, print how many of its test pictures ResNet found, out of how many, and the percent.

Show 12 test pictures, or fewer if needed, chosen with numpy default_rng(42), in a 3 by 4 grid. Use the same seed as the plain model so the photos match. Title each with the guess and the real group. Green text when they match, red when they do not. Figure title: ResNet. Green means the guess matches. Red means it does not. Clip to 0 to 255. plt.show().
Then print:
Green titles are correct guesses. Red titles are mistakes.
This is a classroom model. It must not be used to diagnose a real animal.
```

### You will know it worked when

- You see charts titled "ResNet", with lines named Train and Validation.
- You see a number grid about where ResNet got confused.
- You see 12 test photos. Green titles are right. Red titles are mistakes.
- A sentence gives ResNet's test percent.

## Step 8. Compare the two models

### For you

Both models were scored on the same test pictures. This step puts those two percents side by side. A higher bar means that model was right more often. The dotted line at 33 percent is what guessing would score.

### Copy this prompt

```text
Write one notebook cell. Python only. No explanation, and no lines that start with ! or %.
Do not train, and do not rebuild the datasets.
Needs simple_test_accuracy and resnet_test_accuracy. If one is missing, print "Score both the plain model and ResNet on the test pictures first." and return.

Draw one bar chart titled "Test pictures: which model was right more often".
Two bars: Plain model and ResNet. The heights are simple_test_accuracy and resnet_test_accuracy. Y-axis "Percent correct", from 0 to 100. Write each percent above its bar, rounded to the nearest whole number.
Draw a horizontal dotted line at 33 and label it Guessing.
Call plt.show().

Print: The plain model was correct on about X percent of the test pictures.
Print: ResNet was correct on about Y percent of the test pictures.
Print one more sentence with which model was higher and by how many percentage points, rounded to the nearest whole number. If the gap is under 1 point, say the two models were about the same on the test pictures.
Print: Both scores are classroom results. Neither model is a veterinarian.
```

### You will know it worked when

You see two bars, one for the plain model and one for ResNet, and a sentence that says which test score was higher.

## Step 9, optional. Try one photo

### For you

Use this if you want both models to guess on one more cattle photo. In Colab, run the cell and choose an image file when it asks.

The sentences you get are classroom results, not a diagnosis.

### Copy this prompt

```text
Write one notebook cell. Python only. No explanation, and no lines that start with ! or %.
Do not train or rebuild either model.
Needs simple_model, resnet_model, class_names, and IMG_SIZE. If one is missing, print "Train both models first, then try one photo." and return.

In Google Colab, let the student choose a file with google.colab.files.upload and open it with Pillow.
Otherwise open my_cow.jpg from the current folder.
If no photo is available, print: To try one photo in Colab, run this cell and choose an image file. Outside Colab, put a photo named my_cow.jpg in the notebook folder and run this cell again.

Show the color photo, titled "The photo you gave the models", and call plt.show().
Resize to IMG_SIZE by IMG_SIZE, add a batch dimension, and keep pixels in the 0 to 255 range. Do not divide by 255. Each model already includes its own preprocessing.
Predict that same batch with simple_model and with resnet_model.
For each model, print the top guess, using Foot and mouth, Healthy, and Lumpy skin, and how sure it is, as a percent. Say which sentence is the plain model and which is ResNet.
Then print one sentence per group with that model's percent.
End with: These are classroom guesses. They are not a veterinary diagnosis.
```

### You will know it worked when

You see your photo and two guesses, one from the plain model and one from ResNet.

## When a cell turns red

Copy the red error. In the prompt below, replace `PASTE_THE_RED_ERROR_HERE` with that error. Click the sparkle on that same cell, or paste the prompt into the LLM of your choice, and replace the cell with the new Python. Then run it again. Leave earlier cells alone when they already finished.

### Copy this prompt

```text
The current notebook cell failed. Here is the red error:

PASTE_THE_RED_ERROR_HERE

Rewrite that one cell. Python only.
Keep any of these names that already exist: DATA_DIR, class_names, train_ds, val_ds, test_ds, data_augmentation, IMG_SIZE, BATCH_SIZE, SEED, class_weight, simple_model, simple_history, simple_test_accuracy, resnet_model, resnet_history, resnet_test_accuracy.
Do not redo steps that already worked. Do not use the test set while training. Print short sentences a beginner can read.
```
