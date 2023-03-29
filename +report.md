# HW2 Report - Mike Liu

[//]: # (## Code Structure)

## Train \& Test

### Ablation Study

This code uses [`ablation.json`](ablation.json) file to config the parameters, and experiments to run are specified by network strings. e.g.:

```json
{
  "networks": [
    "DQN_h:t64t64_lr:0.001_e:0.3_ed:1_em:0_tu:100_bs:32_rs:10000",
    "DQN_h:t64t64_lr:0.001_e:0.3_ed:1_em:0_tu:100_bs:32_rs:32",
    "DQN_h:t64t64_lr:0.001_e:0.3_ed:1_em:0_tu:1_bs:32_rs:10000",
    "DQN_h:t64t64_lr:0.001_e:0.3_ed:1_em:0_tu:1_bs:32_rs:32",
    "DQN_h:t64t64_lr:0.001_e:0.3_ed:1_em:0_tu:100_bs:32_rs:10000",
    "DQN_h:t64t64_lr:0.001_e:0.1_ed:1_em:0_tu:100_bs:32_rs:10000",
    "DQN_h:t64t64_lr:0.001_e:0.5_ed:1_em:0_tu:100_bs:32_rs:10000",
    "DQN_h:t32t32t32_lr:0.001_e:1_ed:0.99_em:0.3_tu:100_bs:32_rs:10000"
  ]
}
```

By adding different network string here, the strings will be parsed into network structure and parameters for each experiment.
Take the first line as an example,
```
"DQN_h:t64t64_lr:0.001_e:0.3_ed:1_em:0_tu:100_bs:32_rs:10000"
```
`DQN` is the network name, `h` is the hidden layer, `t64t64` means two hidden layers with 64 neurons each with tanh activation. It supports `t` for tanh activation, `s` for sigmoid activation, `r` for relu activation, `l` for leaky relu activation, `e` for elu activation, etc.
 `lr` is the learning rate, `e` is the epsilon, `ed` is the epsilon decay, `em` is the epsilon minimum, `tu` is the target update, `bs` is the batch size, `rs` is the replay size.

Therefore, this config above will run 8 experiments with different network structures and parameters, and one can add new experiment with minimal effort.

## Results

## Discussion