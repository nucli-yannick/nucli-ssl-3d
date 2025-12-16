from __future__ import annotations

import onnx
import onnx.numpy_helper as numpy_helper
import numpy as np
import matplotlib.pyplot as plt


model = onnx.load('low_high.onnx')

stages = {'stage0' : 'onnx::Conv_1674', 'stage1' : 'onnx::Conv_1668', 'stage2' : 'onnx::Conv_1662', 'stage3' : 'onnx::Conv_1656'}

fig, axs = plt.subplots(4, 2)

for i, (stage, name) in enumerate(stages.items()):

    [weights] = [w for w in model.graph.initializer if w.name == name]
    w = onnx.numpy_helper.to_array(weights)
    a = np.sqrt(np.sum(w**2, axis=(0, 2, 3, 4)))

    axs[i, 1].plot(a)
    axs[i, 1].set_title(f'{stage}: weights in concat convolution')


stages = {'stage0' : 'onnx::Conv_1629', 'stage1' : 'onnx::Conv_1635', 'stage2' : 'onnx::Conv_1641', 'stage3' : 'onnx::Conv_1647'}


for i, (stage, name) in enumerate(stages.items()):

    [weights] = [w for w in model.graph.initializer if w.name == name]
    w = onnx.numpy_helper.to_array(weights)
    a = np.sqrt(np.sum(w**2, axis=(0, 2, 3, 4)))

    axs[i, 0].plot(a)
    axs[i, 0].set_title(f'{stage}: weights in encoder convolution')


plt.show()