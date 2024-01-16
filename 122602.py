import os
import json
import keras
import time
import keras_nlp
import tensorflow as tf
import requests
import joblib
keras.mixed_precision.set_global_policy("float32")
preprocessor = keras_nlp.models.GPT2CausalLMPreprocessor.from_preset(
    "gpt2_base_en",
    sequence_length=128,
)
gpt2_lm = keras_nlp.models.GPT2CausalLM.from_preset(
    "gpt2_base_en", preprocessor=preprocessor
)

# poem_collection = []
paragraphs=[]
# for file in os.listdir("dataset/chinesepoetry"):
#     if ".json" not in file or "poet" not in file:
#         continue
#     full_filename = "%s/%s" % ("dataset/chinesepoetry", file)
#     with open(full_filename, "r") as f:
#         content = json.load(f)
#         poem_collection.extend(content)
for i in range(120):
    
    if(i<9):
        ittext="00{}".format(i+1)
    elif(i<99):
        ittext="0{}".format(i+1)
    else:
        ittext="{}".format(i+1)
    response=requests.get("https://zh.wikisource.org/w/api.php?action=query&format=json&prop=extracts&titles=紅樓夢/第{}回&formatversion=2&explaintext=1&exsectionformat=plain".format(ittext))

    paragraphs.append(response.json()["query"]["pages"][0]["extract"].replace("上一回\u3000回目录\u3000下一回",""))

# paragraphs = ["".join(data["paragraphs"]) for data in poem_collection]
train_ds = (
    tf.data.Dataset.from_tensor_slices(paragraphs)
    .batch(16)
    .cache()
    .prefetch(tf.data.AUTOTUNE)
)

# Running through the whole dataset takes long, only take `500` and run 1
# epochs for demo purposes.
train_ds = train_ds.take(2000)
num_epochs = 10

learning_rate = keras.optimizers.schedules.PolynomialDecay(
    5e-4,
    decay_steps=train_ds.cardinality() * num_epochs,
    end_learning_rate=0.0,
)
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
gpt2_lm.compile(
    optimizer=keras.optimizers.Adam(learning_rate),
    loss=loss,
    weighted_metrics=["accuracy"],
)

gpt2_lm.fit(train_ds, epochs=num_epochs)
joblib.dump(gpt2_lm,"20231226badgpt.pkl")
def run(arug):
    start = time.time()

    output = gpt2_lm.generate(arug, max_length=200)
    print("\nGPT-2 output:")
    print(output)

    end = time.time()
    print(f"TOTAL TIME ELAPSED: {end - start:.2f}s")
while(True):
    arug=input("What are you asking for?")
    if(arug=="-1"):
        break
    else:
        run(arug)