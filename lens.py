import torch
from tuned_lens.nn.lenses import TunedLens, LogitLens
from transformers import AutoModelForCausalLM, AutoTokenizer
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tuned_lens.plotting import PredictionTrajectory
from transformers import GPT2LMHeadModel, GPT2Tokenizer

device = torch.device('cpu')

model_name = "gpt2"  # You can use other GPT-2 variants like "gpt2-medium", "gpt2-large", etc.
model = GPT2LMHeadModel.from_pretrained(model_name)
model = model.to(device)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

tuned_lens = TunedLens.from_model_and_pretrained(model, map_location=device)
tuned_lens = tuned_lens.to(device)

unembed_method = tuned_lens.unembed
logit_lens = LogitLens(unembed=unembed_method)

input_ids_ring = tokenizer.encode(
    "the world war 2 started on 1938 and ended on "
)

targets_ring = input_ids_ring[1:] + [tokenizer.eos_token_id]

line = slice(0, len(input_ids_ring))


# Create PredictionTrajectory instances for TunedLens and LogitLens
prediction_traj_tuned = PredictionTrajectory.from_lens_and_model(
    tuned_lens,
    model,
    tokenizer=tokenizer,
    input_ids=input_ids_ring,
    targets=targets_ring,
).slice_sequence(line)

prediction_traj_logit = PredictionTrajectory.from_lens_and_model(
    logit_lens,
    model,
    tokenizer=tokenizer,
    input_ids=input_ids_ring,
    targets=targets_ring,
).slice_sequence(line)

# Create subplots for both lenses
fig = make_subplots(
    rows=4,
    cols=2,
    shared_xaxes=True,
    vertical_spacing=0.05,
    subplot_titles=("Tuned Lens - Entropy", "Logit Lens - Entropy",
                    "Tuned Lens - Forward KL", "Logit Lens - Forward KL",
                    "Tuned Lens - Cross Entropy", "Logit Lens - Cross Entropy",
                    "Tuned Lens - Max Probability", "Logit Lens - Max Probability"),
)

# Add traces for Tuned Lens
fig.add_trace(
    prediction_traj_tuned.entropy().heatmap(
        colorbar_y=0.89, colorbar_len=0.25, textfont={'size': 10}
    ),
    row=1, col=1
)

fig.add_trace(
    prediction_traj_tuned.forward_kl().heatmap(
        colorbar_y=0.63, colorbar_len=0.25, textfont={'size': 10}
    ),
    row=2, col=1
)

fig.add_trace(
    prediction_traj_tuned.cross_entropy().heatmap(
        colorbar_y=0.37, colorbar_len=0.25, textfont={'size': 10}
    ),
    row=3, col=1
)

fig.add_trace(
    prediction_traj_tuned.max_probability().heatmap(
        colorbar_y=0.11, colorbar_len=0.25, textfont={'size': 10}
    ),
    row=4, col=1
)

# Add traces for Logit Lens
fig.add_trace(
    prediction_traj_logit.entropy().heatmap(
        colorbar_y=0.89, colorbar_len=0.25, textfont={'size': 10}
    ),
    row=1, col=2
)

fig.add_trace(
    prediction_traj_logit.forward_kl().heatmap(
        colorbar_y=0.63, colorbar_len=0.25, textfont={'size': 10}
    ),
    row=2, col=2
)

fig.add_trace(
    prediction_traj_logit.cross_entropy().heatmap(
        colorbar_y=0.37, colorbar_len=0.25, textfont={'size': 10}
    ),
    row=3, col=2
)

fig.add_trace(
    prediction_traj_logit.max_probability().heatmap(
        colorbar_y=0.11, colorbar_len=0.25, textfont={'size': 10}
    ),
    row=4, col=2
)

# Update the layout and display the plot
fig.update_layout(height=800, title_text="GPT2-xl Tuned vs Logit Lens Comparison", template="plotly", margin=dict(l=0, r=0, t=90, b=0))
fig.show()
