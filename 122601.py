# /*
#  * ----------------------------------------------------------------------------
#  * "THE TEA-WARE LICENSE" (Revision 42-1):
#  * <210114@gs.hs.ntnu.edu.tw> wrote this file.  As long as you retain this notice you
#  * can do whatever you want with this stuff. If we meet some day, and you think
#  * this stuff is worth it, you can buy me a tea (tea or beer both fine) in return.   Lapsang Souchong
#  * Please notice that some part of the code might not be written by me.
#  * File: 122601.py
#  * ----------------------------------------------------------------------------
#  */
#

# import os
#
# os.environ["KERAS_BACKEND"] = "tensorflow"  # or "tensorflow" or "torch"
#!pip install git+https://github.com/keras-team/keras-nlp.git
import keras_nlp
import keras
import tensorflow as tf
import time

keras.mixed_precision.set_global_policy("float32")

# To speed up training and generation, we use preprocessor of length 128
# instead of full length 1024.
preprocessor = keras_nlp.models.GPT2CausalLMPreprocessor.from_preset(
    "gpt2_base_en",
    sequence_length=128,
)
gpt2_lm = keras_nlp.models.GPT2CausalLM.from_preset(
    "gpt2_base_en", preprocessor=preprocessor
)

def run(arug):
    start = time.time()

    output = gpt2_lm.generate(arug, max_length=200)
    print("\nGPT-2 output:")
    print(output)

    end = time.time()
    print(f"TOTAL TIME ELAPSED: {end - start:.2f}s")
while(True):
    arug=input("What are you asking for?")
    if(int(arug)==-1):
        break
    else:
        run(arug)