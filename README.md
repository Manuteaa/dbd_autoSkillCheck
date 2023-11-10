# DBD Auto Skill Check

The Dead by Daylight Auto Skill Check is a tool developed using deep learning techniques (PyTorch) to automatically detect and successfully hit skill checks in the popular game Dead by Daylight. 
This tool is designed to improve gameplay performance and enhance the player's skill in the game. 

## Features
- Real-time detection of skill checks
- High accuracy in recognizing skill checks
- Automatic triggering of great skill checks through auto-pressing the space bar


## What is a skill check

A skill check is a game mechanic in Dead by Daylight that allows the player to progress faster in a specific action such as repairing generators or healing teammates.
It occurs randomly and requires players to press the space bar to stop the progression of a red cursor.

Skill checks can be: 
- failed, if the cursor misses the designated white zone
- successful, if the cursor lands in the white zone 
- or greatly successful, if the cursor accurately hits the white-filled zone 

Here are examples of different great skill checks:

|              Great skill check (type 1)              |              Great skill check (type 2)              |              Great skill check (type 3)              |
|:----------------------------------------------------:|:----------------------------------------------------:|:----------------------------------------------------:|
| ![](tests/data/2/20230617-140530_11483.png "Type 1") | ![](tests/data/2/20230617-140530_39896.png "Type 2") | ![](tests/data/2/20230617-142505_22039.png "Type 2") |

Successfully hitting a skill check increases the speed of the corresponding action, and a greatly successful skill check provides even greater rewards. 
On the other hand, missing a skill check reduces the action's progression speed and alerts the ennemi with a loud sound.

## Dataset
We designed a custom dataset from in-game screen recordings and frame extraction of gameplay videos on youtube.
To save disk space, we center-crop each frame to size 320x320 before saving.

The data was manually divided into three separate folders based on the following categories:
- Images without skill check
- Images with a skill check on hold (the skill check should not be hit yet)
- Images with a skill check to hit

To alleviate the laborious collection task, we employed data augmentation techniques such as random rotations, random crop-resize, and random brightness/contrast/saturation adjustments.

We developed a customized and optimized dataloader that automatically parses the dataset folder and assigns the correct label to each image based on its corresponding folder.
Our data loaders use a custom sampler to handle imbalanced data.

## Architecture
The skill check detection system is based on an encoder-decoder architecture. 

The encoder employs the MobileNet V3 Large architecture, specifically chosen for its trade-off between inference speed and accuracy. 
This ensures real-time inference and quick decision-making without compromising detection precision.
We also compare the architecture with the MNASNet 0.5 architecture, which is even faster than MobileNet V3 Large but with a slight decrease in accuracy.

The decoder is a simple MLP with a single hidden layer. The MLP predicts three logits, providing the valuable information for the decision-making.

## Training

We use a standard cross entropy loss to train the model.


## Inference
We provide a script that loads the trained model and monitors the main screen.
For each sampled frame, the script will center-crop and normalize the image then feed it to the model.

When a skill check is detected, the script will automatically press the space bar to trigger the great skill check, 
then it waits for a short period of time (2s) to avoid triggering the same skill check multiple times in a row.

To achieve real time results, we convert the model to ONNX format and use the ONNX runtime to perform inference. 
We observed a 1.5x to 2x speedup compared to baseline inference.

## Results

We test our model on a real in-game session, and we obtain the following results:

|          Encoder architecture           | Great skill check precision | Great skill check recall | # Total params | In game average inference time (onnx) |
|:---------------------------------------:|:---------------------------:|:------------------------:|:--------------:|:-------------------------------------:|
|           MobileNet v3 Large            |            99.1%            |          99.6%           |      6.5M      |                 10ms                  |
|              MNASNet  0.5               |            98.3%            |          99.4%           |      3.2M      |                  2ms                  |


In conclusion, our model achieves high accuracy thanks to the high-quality dataset with effective data augmentation techniques, 
and architectural choices.

During our laptop testing, we observed rapid inference times of approximately 10ms per frame using MobileNet V3 Large 
and an even faster 2ms per frame with MNASNet 0.5. When combined with our screen monitoring script, 
we achieved a consistent 60fps detection rate, which is enough for real-time detection capabilities.

However, we encountered an unplanned challenge during actual gameplay. Despite successfully detecting and reacting to the great skill check 
within a response time of 2ms to 10ms, we faced inherent latency issues commonly found in multiplayer games.
Consequently, despite the quick decisions of the model, we still missed the great skill check.

To address this issue, we propose a solution: anticipating and hitting the great skill check a few frames in advance. 
For instance, we can develop a model that estimates the time it takes for the red cursor to reach the great white zone. 
Following variables such as the user's network latency to the game server, we can hit the space bar preemptively. 
This approach aligns with the natural behavior of real players who intuitively anticipate and time their actions based on the red cursor's movement speed. 
We will implement this strategy in future work.
