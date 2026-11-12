# Vibecoding a cattle photo model

Vibecoding means you describe what you want in ordinary English, and an AI helper writes the code.

You will teach a computer to look at a photo of cattle and place it in one of three groups:

1. **Healthy**
2. **Lumpy skin**
3. **Foot and mouth**

You will train two models on the same photos, then compare them.

1. A **plain model**. It starts knowing nothing about photos and learns only from these cattle pictures.
2. **ResNet**. It already knows how to look at ordinary photos. You teach it these three groups.

The photos come from a public collection, the [Cattle diseases dataset](https://www.kaggle.com/datasets/devang03mgr/cattle-diseases-datasets). Register a Kaggle account so you can download the zip.

You will not write the program yourself. This page has the instructions. Each gray box is written the way a beginner would ask, with no library names and no settings to memorize. Copy that prompt, then either click the sparkle in Colab or paste it into the LLM of your choice. The helper writes the Python. You paste that Python into Colab and run it.

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
| Validation | Photos used during teaching to see if it is improving. About 15 percent |
| Test | Photos kept aside until the end. The final score uses only these. About 15 percent |
| Epoch | One pass through the training photos |
| Plain model | The first model. It learns only from these cattle pictures |
| ResNet | The second model. It already knows how to look at ordinary photos |

The zip is about 270 megabytes. Each teaching step often takes 10 to 25 minutes. Leave the tab open while a cell runs.

## Step 1. Download the pictures

### For you

The zip contains a folder named `Cows datasets`. Inside it are three folders of photos: `foot-and-mouth`, `healthy`, and `lumpy`. The folder name is the group.

The prompt already looks for `/content/archive.zip`. Change that line only if your file is somewhere else, or if you have a direct download link.

### Copy this prompt

```text
Hi. I put a zip of cattle photos in this notebook at:

/content/archive.zip

Can you unpack it for me?

The photos are already sorted into folders. The folder name is the answer, like which disease it is. There should be three folders inside a folder called Cows datasets: foot-and-mouth, healthy, and lumpy. That folder name has a space in it.

Please tell me in normal sentences where the photos are, and how many photos are in each group. Put commas in the big numbers.

If I left a web link there instead of a file, download it. Google Drive and Dropbox links are fine. If the file is missing, or it is a webpage instead of a zip, just tell me what went wrong in one sentence. Don't train anything yet.
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
Before we teach anything, help me look at the cattle photos from the last step.

Tell me how many groups there are and how many photos in total. Tell me how many are in Foot and mouth, Healthy, and Lumpy skin, and which group is biggest and smallest. The photos are not all the same size, so also tell me the width and height of two photos from each group.

Please say that the folder name is the real answer, and that the file names can be messy so I should ignore them. Remind me to look at the zoom, the background, and the lighting.

Then show me a bar chart called "How many pictures are in each group", with the count written on each bar. Also show me 3 example photos from each group, with the group name on them. I want to see the pictures, not a list of file names.
```

### You will know it worked when

You see the same three counts, a bar chart, and three example photos from each group. The writing on the screen is ordinary English.

## Step 3. Prepare the train, validation, and test pictures

### For you

Every photo has to be the same size. The helper then splits them into three piles:

- **Train**, about 70 percent. The model learns from these.
- **Validation**, about 15 percent. During teaching, the model is scored on these so it can stop when it stops improving.
- **Test**, about 15 percent. Training does not use these. Both models are scored on this same pile, so the comparison is fair.

Each disease should show up in all three piles. Training photos can be flipped or tilted a little. The animal is still the same animal. Validation photos and test photos stay as they are.

### Copy this prompt

```text
The cattle photos are different sizes. Please get them ready so a model can learn from them. Use the photos we already unpacked. Don't download them again, and don't start training yet.

Make every photo the same size and keep them in color.

Split them into three piles, and do it separately for each disease so every disease shows up in all three:
- about 70 percent to train on
- about 15 percent to check progress while training (validation)
- about 15 percent to save for the real test at the end

A photo should only be in one pile. Please don't mix the test photos back in later. We are not allowed to look at the test photos while the model is learning.

It is ok to flip or slightly tilt the training photos so the model does not memorize one exact view. Leave the validation photos and the test photos alone.

Show me 5 slightly changed versions of one training photo, so I can see that it is still the same cow. Tell me in sentences how the photos were split, and how many batches of training, validation, and test photos there are. A batch is a small stack of about 32 photos.
```

### You will know it worked when

You see the three group names, a count of training batches, validation batches, and test batches, and one training photo repeated five times with small differences.

## Step 4. Train the plain model

### For you

This first model starts knowing nothing about photos. It learns only from the cattle pictures. It is not ResNet, and it is not a model that was already trained on other photos.

It learns from the training photos. While it learns, it is scored on the validation photos. It must not look at the test photos yet.

The groups are not the same size. The smaller groups should still get a fair chance. This model and the later ResNet model should both get about 5 passes, so the comparison stays fair.

Leave the cell running. A moving progress bar means it is working. This often takes 10 to 25 minutes on a GPU. It may finish early if the validation score stops improving.

### Copy this prompt

```text
Please train a simple photo model on the cattle pictures we already prepared. I don't know the technical settings, so please choose sensible ones and keep going.

This model should start from scratch. Please don't use ResNet, and don't use a model that already learned from other photos. It should learn only from these cattle pictures.

Let it learn for about 5 passes through the training photos. While it learns, check it on the validation photos. If it stops improving on those, you can stop early and keep the best version. Do not look at the test photos during this step.

Some diseases have fewer photos than others. Please don't let the model ignore the smaller groups.

Save the trained model, and tell me in a normal sentence what percent it got right on the validation photos. Also say that this is for learning and it is not a veterinarian.

A progress bar is good. Tell me when training starts, because it can take a while.
```

### You will know it worked when

The cell prints a validation percent and says the model was saved.

A score above 33 percent is better than guessing. The next step is how it does on the test photos.

## Step 5. See how the plain model did

### For you

This step scores the plain model on the test photos, which it was not allowed to use while learning.

You will see how the scores changed over time, which groups got mixed up, and some test photos with the guess next to the real group. Green means right. Red means wrong. Keep the test percent. Step 8 compares it with ResNet.

### Copy this prompt

```text
The simple cattle model is trained. Please show me how it did, in a way a beginner can read. Don't train it again.

First, chart how it did while learning. One chart for how often it was right, and one for how far off it was. Show the training photos and the validation photos on both charts. Call the lines Train and Validation.

Then score it on the test photos only, the ones it never learned from. Tell me the percent it got right. Say that guessing one of the three groups would be right about 33 percent of the time.

Show me a grid of counts for where it mixed the groups up. Title it something like "Plain model: where it got confused on the test pictures." The rows should be the real group and the columns should be the guess. Put the count in each square. Tell me which mix-up happened most, in one sentence. Also tell me, for each disease, how many of its test photos the model found.

Show 12 test photos. On each one, write the guess and the real group. Use green text when it is right and red text when it is wrong.

Remember this test percent. We will compare it with another model later.

End by saying green titles are correct, red titles are mistakes, and this must not be used to diagnose a real animal.
```

### You will know it worked when

- You see charts with lines named Train and Validation.
- You see a number grid about where the plain model got confused.
- You see 12 test photos. Green titles are right. Red titles are mistakes.
- A sentence gives the plain model's test percent.

## Step 6. Train ResNet

### For you

ResNet already knows how to look at ordinary photos. This step keeps that skill and teaches it the three cattle groups. The first run downloads that starter model.

Use the same photos as the plain model. Same training photos, same validation photos, and about 5 passes. Do not look at the test photos while it learns.

Leave the cell running. A moving progress bar means it is working. This often takes 10 to 25 minutes on a GPU.

### Copy this prompt

```text
Now please train a second model, using the same cattle photos we already split. Don't redo the split, and don't look at the test photos while this one is learning.

This time, start from ResNet, the pretrained one that already knows how to look at normal photos. Keep what it already knows, and teach it our three groups: Foot and mouth, Healthy, and Lumpy skin. Don't train a brand new model from scratch, and don't replace the simple model we already saved.

Give it about 5 passes through the training photos, same idea as the first model. Check it on the validation photos while it learns, and keep the best version if it stops improving. The smaller disease groups should still get a fair chance.

The first time you do this, it may need to download the starter ResNet. That is ok.

Save this model separately from the first one. Tell me in a sentence what percent it got right on the validation photos, and say it is for learning and not a veterinarian.
```

### You will know it worked when

The cell prints a validation percent and says the ResNet model was saved.

The next step scores it on the same test photos the plain model already saw.

## Step 7. See how ResNet did

### For you

This is the same kind of score as Step 5, on the same test photos, for ResNet. Keep this test percent. The next step puts it next to the plain model.

### Copy this prompt

```text
Please show me how the ResNet cattle model did, the same way you showed the simple model. Don't train it again, and don't change the simple model's score.

Chart how often it was right, and how far off it was, for the training photos and the validation photos. Title the charts so I can tell they are the ResNet ones.

Then score it only on the test photos. These should be the same test photos used for the simple model. Tell me the percent it got right, and remind me that guessing would be about 33 percent.

Show the mix-up grid for ResNet, say which mix-up happened most, and say how many test photos it found for each disease.

Show 12 test photos with the guess and the real group. Green if it is right, red if it is wrong. If you can, use the same 12 photos you showed for the simple model, so I can compare.

Remember this test percent too.

End by saying this is a classroom model and must not be used to diagnose a real animal.
```

### You will know it worked when

- You see charts for ResNet, with lines named Train and Validation.
- You see a number grid about where ResNet got confused.
- You see 12 test photos. Green titles are right. Red titles are mistakes.
- A sentence gives ResNet's test percent.

## Step 8. Compare the two models

### For you

Both models were scored on the same test photos. This step puts those two percents side by side. A higher bar means that model was right more often. The line at 33 percent is what guessing would score.

### Copy this prompt

```text
We now have two cattle models, the simple one and the ResNet one. Both were scored on the same test photos. Please compare them for me. Don't train anything.

Make a bar chart called "Test pictures: which model was right more often." One bar for the plain model and one bar for ResNet. Show percent correct from 0 to 100, and write the percent on each bar. Add a dotted line at 33 percent labeled Guessing, since a random guess among three groups would be about that.

Then tell me, in a few sentences, the plain model's test percent, ResNet's test percent, and which one did better and by about how much. If they are basically tied, say that.

Say that both scores are classroom results and neither model is a veterinarian.
```

### You will know it worked when

You see two bars, one for the plain model and one for ResNet, and a sentence that says which test score was higher.

## Step 9, optional. Try one photo

### For you

Use this if you want both models to guess on one more cattle photo. In Colab, run the cell and choose an image file when it asks.

The sentences you get are classroom results, not a diagnosis.

### Copy this prompt

```text
I want to try one new cattle photo on both models we trained. Don't train them again.

If this is Google Colab, let me pick a photo from my computer. If not, look for a file named my_cow.jpg. If you can't find a photo, tell me how to provide one.

Show me the photo. Then tell me what the plain model thinks it is, and how sure it is. Then tell me what ResNet thinks, and how sure it is. Use the names Foot and mouth, Healthy, and Lumpy skin. Also tell me the percent for each group.

End by saying these are classroom guesses, not a veterinary diagnosis.
```

### You will know it worked when

You see your photo and two guesses, one from the plain model and one from ResNet.

## When a cell turns red

Copy the red error. In the prompt below, replace `PASTE_THE_RED_ERROR_HERE` with that error. Click the sparkle on that same cell, or paste the prompt into the LLM of your choice, and replace the cell with the new Python. Then run it again. Leave earlier cells alone when they already finished.

### Copy this prompt

```text
This cell crashed. Here is the red error:

PASTE_THE_RED_ERROR_HERE

Can you fix just this cell? Please don't start the whole lesson over, and don't redo the steps that already worked. Don't peek at the test photos if this cell is a training step. Talk to me in short sentences.
```
