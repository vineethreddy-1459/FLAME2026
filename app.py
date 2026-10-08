import json

import numpy as np
import streamlit as st


st.set_page_config(
    page_title="Live Neural Network",
    page_icon="🧠",
    layout="centered"
)


@st.cache_data
def load_model():
    with open(
        "model.json",
        encoding="utf-8"
    ) as f:
        return json.load(f)


model = load_model()

hidden_w = np.array(
    model["hidden"]["weights"],
    dtype=np.float32
)

hidden_b = np.array(
    model["hidden"]["biases"],
    dtype=np.float32
)

output_w = np.array(
    model["output"]["weights"],
    dtype=np.float32
)

output_b = float(
    model["output"]["bias"]
)


def forward_pass(x):
    hidden_z = (
        float(x) * hidden_w
        + hidden_b
    )

    hidden_a = np.maximum(
        0,
        hidden_z
    )

    prediction = float(
        np.dot(
            hidden_a,
            output_w
        )
        + output_b
    )

    return hidden_z, hidden_a, prediction


st.title("🧠 Live Neural Network")

st.write(
    "Move the input and watch the "
    "trained neurons process it."
)

st.markdown(
    "**Celsius → Hidden Layer → Output → Fahrenheit**"
)

x = st.slider(
    "Input — Celsius",
    min_value=-20.0,
    max_value=100.0,
    value=25.0,
    step=0.5
)

hidden_z, hidden_a, prediction = (
    forward_pass(x)
)

st.subheader("1. Input")

st.metric(
    "Celsius",
    f"{x:.1f} °C"
)

st.subheader("2. Hidden Layer")

cols = st.columns(3)

for i, col in enumerate(cols):

    with col:

        st.markdown(
            f"### Neuron {i + 1}"
        )

        st.write(
            f"Weighted sum: "
            f"`{hidden_z[i]:.2f}`"
        )

        st.metric(
            "Activation",
            f"{hidden_a[i]:.2f}"
        )

        st.caption(
            f"Weight = {hidden_w[i]:.3f}"
        )

        st.caption(
            f"Bias = {hidden_b[i]:.3f}"
        )

st.subheader("3. Output Layer")

st.metric(
    "Predicted Fahrenheit",
    f"{prediction:.2f} °F"
)

st.info(
    "Input → weight × input + bias → "
    "ReLU → output neuron → prediction"
)

with st.expander(
    "Show learned parameters"
):

    st.write(
        "Hidden weights:",
        hidden_w.tolist()
    )

    st.write(
        "Hidden biases:",
        hidden_b.tolist()
    )

    st.write(
        "Output weights:",
        output_w.tolist()
    )

    st.write(
        "Output bias:",
        output_b
    )
