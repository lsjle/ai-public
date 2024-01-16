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