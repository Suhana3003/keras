import os
import sys
import time
import logging
import gc
import psutil
import requests
import numpy as np
import tensorflow as tf
import keras
import keras_hub
from transformers import AutoTokenizer  # Hugging Face tokenizer

# -------------------- ENV SETUP --------------------
print("🚀 Setting up environment (CPU JAX)...")

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"  # must be set before jax import

from keras.src.distribution import distribution_lib
import jax
import jax.numpy as jnp

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# -------------------- JAX DEVICES --------------------
DEVICES = jax.device_count()
if DEVICES < 2:
    print("🛑 Need >=2 devices for AutoShardDistribution (set XLA_FORCE_HOST_PLATFORM_DEVICE_COUNT=2).")
    sys.exit(1)

print(f"✅ JAX Devices Detected: {DEVICES}")

# -------------------- MODEL CONFIG --------------------
# Choose your model: "opt_125m_en" or "gpt2_base_en"
MODEL_PRESET = "opt_125m_en"  # change to "gpt2_base_en" to train GPT2

SEQUENCE_LENGTH = 128
BATCH_SIZE = DEVICES * 4
STEPS_PER_EPOCH = 50
EPOCHS = 10

# Map presets to keras_hub classes
MODEL_MAPPING = {
    "opt_125m_en": keras_hub.models.OPTCausalLM,
    "gpt2_base_en": keras_hub.models.GPT2CausalLM,
}
MODEL_CLASS = MODEL_MAPPING[MODEL_PRESET]

# Hugging Face tokenizer mapping
TOKENIZER_MAPPING = {
    "opt_125m_en": "facebook/opt-125m",
    "gpt2_base_en": "gpt2",
}
hf_tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MAPPING[MODEL_PRESET])

def tokenize_text(text):
    enc = hf_tokenizer(
        text,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=SEQUENCE_LENGTH + 1,
    )
    return enc["input_ids"][0]

# -------------------- MEMORY HELPERS --------------------
MAX_RSS_MB = 0

def get_rss_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**2

def log_mem(stage):
    global MAX_RSS_MB
    cur = get_rss_mb()
    MAX_RSS_MB = max(MAX_RSS_MB, cur)
    print(f"  {stage} Memory: {cur:.2f} MB (peak {MAX_RSS_MB:.2f} MB)")
    return cur, MAX_RSS_MB

# -------------------- DATASET --------------------
def load_dataset():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    try:
        text = requests.get(url, timeout=5).text
    except Exception:
        text = "To be, or not to be, that is the question."

    # Tokenize full text
    tokens = hf_tokenizer(text, return_tensors="np")["input_ids"].flatten()
    tokens = np.array(tokens, dtype=np.int32)

    n = (len(tokens) // (SEQUENCE_LENGTH + 1)) * (SEQUENCE_LENGTH + 1)
    if n == 0:
        raise RuntimeError("Not enough tokens for even one training sequence.")

    seqs = tokens[:n].reshape(-1, SEQUENCE_LENGTH + 1)

    def gen():
        for seq in seqs:
            x = {
                "token_ids": seq[:-1],
                "padding_mask": np.ones(SEQUENCE_LENGTH, dtype=bool),
            }
            y = seq[1:]
            yield x, y

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            {
                "token_ids": tf.TensorSpec([SEQUENCE_LENGTH], tf.int32),
                "padding_mask": tf.TensorSpec([SEQUENCE_LENGTH], tf.bool),
            },
            tf.TensorSpec([SEQUENCE_LENGTH], tf.int32),
        ),
    )

    return ds.batch(BATCH_SIZE, drop_remainder=True)

# -------------------- TRAINING --------------------
def train_model():
    print(f"\n🌐 Training {MODEL_PRESET} with AutoShard on {DEVICES} CPU devices")
    mesh = distribution_lib.DeviceMesh((DEVICES,), ("batch",))
    dist = distribution_lib.AutoShardDistribution(mesh)

    log_mem("START")

    with dist.scope():
        model = MODEL_CLASS.from_preset(MODEL_PRESET, preprocessor=None)

    # Warmup shard planning
    sample_batch = next(iter(load_dataset().take(1)))
    sample_x, _ = sample_batch
    from jax import tree_util
    sample_x_jax = tree_util.tree_map(jnp.asarray, sample_x)
    dist.shard(model, sample_x_jax)

    model.compile(
        optimizer="adam",
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    )

    train_ds = load_dataset().repeat()

    total_tokens = BATCH_SIZE * SEQUENCE_LENGTH * STEPS_PER_EPOCH * EPOCHS
    print(f"🔄 Training {EPOCHS} epochs × {STEPS_PER_EPOCH} steps")

    log_mem("PRE-FIT")
    start = time.time()
    hist = model.fit(train_ds, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, verbose=1)
    end = time.time()
    log_mem("END")

    ttime = end - start
    tps = total_tokens / ttime
    print("\n📊 Report:")
    print(f"  Time: {ttime:.2f}s")
    print(f"  Throughput: {tps:.2f} tokens/s")
    print(f"  Peak Memory: {MAX_RSS_MB:.2f} MB")
    print(f"  Final Loss: {hist.history['loss'][-1]:.4f}")

if __name__ == "__main__":
    train_model()
