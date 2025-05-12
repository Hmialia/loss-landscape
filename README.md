# Visualizing the Loss Landscape of Neural Nets

这个代码库是用于实现论文 [*Visualizing the Loss Landscape of Neural Nets*](https://arxiv.org/abs/1712.09913)（《可视化神经网络的损失函数》）的PyTorch代码。

**核心功能**

该工具可以帮助研究者和开发者理解神经网络的损失函数形态。具体来说，它可以：

* **计算和可视化损失函数表面**: 在给定网络结构和预训练参数后，该工具可以计算并可视化在最优参数附近的随机方向上的损失函数形态。
* **支持并行计算**: 计算过程可以在单个节点的多个GPU上并行进行，也可以在多个节点上并行进行。
* **存储结果**: 计算得到的随机方向和损失函数值会以HDF5 (`.h5`)文件的格式存储下来。

**可视化类型**

该代码库支持多种可视化方式：

* **1D线性插值 (1D linear interpolations)**: 评估在同一网络损失函数的两个最小值点之间的方向上的损失值。这种方法可以用于比较不同批量大小训练出的最小值点的平坦度。相关脚本是 `plot_surface.py`。
* **1D随机方向图 (plots along random normalized directions)**: 沿着与模型参数维度相同的随机方向（经过“滤波器归一化”处理）采样并绘制损失值。相关脚本是 `plot_surface.py`。也可以使用 `plot_1D.py` 自定义1D图的外观。
* **2D损失等高线图 (2D loss contours)**: 选择两个随机方向并进行归一化处理，然后绘制损失等高线图。相关脚本是 `plot_surface.py`。也可以使用 `plot_2D.py` 自定义等高线图的外观。
* **3D损失表面图 (3D loss surface)**: `plot_2D.py` 可以使用 `matplotlib` 生成基本的3D损失表面图。如果需要更精细的、带光照效果的渲染，可以使用 [ParaView](http://paraview.org)。需要先使用 `h52vtp.py` 将 `.h5` 文件转换为 `.vtp` 文件。

**使用说明**

* **环境要求**: 需要安装PyTorch 0.4, openmpi 3.1.2, mpi4py 2.0.0, numpy 1.15.1, h5py 2.7.0, matplotlib 2.0.2, 和 scipy 0.19。
* **预训练模型**: 代码接受CIFAR-10数据集的预训练PyTorch模型。模型文件应包含 `state_dict`。
* **数据预处理**: 可视化时使用的数据预处理方法应与模型训练时一致。在计算损失值时不使用数据增强（如随机裁剪或水平翻转）。

**主要脚本文件**

* `plot_surface.py`: 用于生成1D和2D的损失表面图。
* `plot_1D.py`: 用于自定义1D损失曲线图的外观。
* `plot_2D.py`: 用于自定义2D损失等高线图和生成基本的3D表面图。
* `h52vtp.py`: 用于将 `.h5` 文件转换为 ParaView 可读的 `.vtp` 文件，以实现更精细的3D可视化。
* `projection.py`: 用于将模型或多个模型投影到由给定方向张成的平面上。
* `net_plotter.py`: 用于操作网络参数和设置带有归一化的随机方向。
* `scheduler.py`: 一个任务调度器，用于将未完成的作业分配给不同的工作单元。
* `plot_trajectory.py`: 用于在由主方向张成的空间中绘制优化路径。
* `plot_hessian_eigen.py`: 用于计算投影曲面的Hessian矩阵及其特征值。

该代码库提供了一个强大的工具集，用于深入探索和理解神经网络的复杂损失函数空间。

---
# 快速开始
## 模型训练
```bash
python -m cifar10.main \
    --model resnet20 \
    --epochs 200 \
    --lr 0.05 \
    --batch_size 64 \
    --optimizer sgd \
    --weight_decay 0.0001 \
    --save_epoch 5
```
## 绘制优化轨迹
``` bash
python plot_trajectory.py \
    --dataset cifar10 \
    --model resnet20 \
    --model_folder /content/loss-landscape/trained_nets/resnet20_sgd_lr=0.05_bs=64_wd=0.0001_mom=0.9_save_epoch=5 \
    --start_epoch 0 \
    --max_epoch 25 \
    --save_epoch 5 \
    --prefix model_ \
    --suffix .t7
```
## 将轨迹叠加在loss面上
```bash
python plot_surface.py \
    --dataset cifar10 \
    --model resnet20 \
    --model_file /content/loss-landscape/trained_nets/resnet20_sgd_lr=0.05_bs=64_wd=0.0001_mom=0.9_save_epoch=5/model_25.t7 \
    --dir_file /content/loss-landscape/trained_nets/resnet20_sgd_lr=0.05_bs=64_wd=0.0001_mom=0.9_save_epoch=5/PCA_weights_save_epoch=5/directions.h5 \
    --x=-5:60:6 \
    --y=-5:25:6 \
    --dir_type weights \
    --loss_name crossentropy \
    --mpi \
    --cuda \
    --plot \
    --proj_file /content/loss-landscape/trained_nets/resnet20_sgd_lr=0.05_bs=64_wd=0.0001_mom=0.9_save_epoch=5/PCA_weights_save_epoch=5/directions.h5_proj_cos.h5 \
    --vmin 0 --vmax 5 --vlevel 0.5
```

---

## plot_trajectory.py

这个 `plot_trajectory.py` 脚本（根据您提供的代码内容判断）通过命令行参数来获取运行所需的信息。这些参数是在代码的 `if __name__ == '__main__':` 部分通过 `argparse` 模块定义的。

以下是这个脚本需要的主要参数：

**必需或核心参数 (通常需要用户提供或具有合理的默认值):**

* **`--model_folder`**: (类型: `str`, 默认: `''`)
    * **作用**: 指定包含一系列模型检查点文件的文件夹路径。这些检查点代表了模型在训练过程中的不同状态，将用于构建优化轨迹。
    * **重要性**: 这是定位模型训练历史的核心参数。

**可选但常用的参数 (用于控制模型的加载和轨迹的生成):**

* **`--dataset`**: (类型: `str`, 默认: `'cifar10'`)
    * **作用**: 指定模型是在哪个数据集上训练的。这会影响模型加载的方式。
* **`--model`**: (类型: `str`, 默认: `'resnet56'`)
    * **作用**: 指定要加载的神经网络模型的名称。脚本会根据这个名称和数据集来实例化正确的模型结构。
* **`--prefix`**: (类型: `str`, 默认: `'model_'`)
    * **作用**: 模型检查点文件名的前缀。例如，如果文件名为 `model_10.t7`，则前缀是 `model_`。
* **`--suffix`**: (类型: `str`, 默认: `'.t7'`)
    * **作用**: 模型检查点文件名的后缀。例如，如果文件名为 `model_10.t7`，则后缀是 `.t7`。
* **`--start_epoch`**: (类型: `int`, 默认: `0`)
    * **作用**: 定义从哪个 epoch 开始收集模型检查点。
* **`--max_epoch`**: (类型: `int`, 默认: `300`)
    * **作用**: 定义收集模型检查点到哪个 epoch 结束。最终的模型（`last_model_file`）也是基于这个 `max_epoch` 来确定的。
* **`--save_epoch`**: (类型: `int`, 默认: `1`)
    * **作用**: 指定每隔多少个 epoch 收集一个模型检查点。例如，如果 `start_epoch=0`, `max_epoch=10`, `save_epoch=2`，则会收集 epoch 0, 2, 4, 6, 8, 10 的模型。
* **`--dir_type`**: (类型: `str`, 默认: `'weights'`)
    * **作用**: 指定在进行PCA和投影时，是基于模型的权重 (`'weights'`) 还是完整状态 (`'states'`, 包括BN层的统计量)。
    * **选项**:
        * `weights`: 只考虑模型的权重和偏置（可能会被 `--ignore` 参数进一步筛选）。
        * `states`: 考虑模型 `state_dict()` 中的所有参数，包括BN层的 `running_mean` 和 `running_var`。
* **`--ignore`**: (类型: `str`, 默认: `''`)
    * **作用**: 在计算PCA方向时，指定要忽略的参数类型。
    * **常见值**: `'biasbn'` 表示忽略偏置（bias）和批量归一化（Batch Normalization）层的参数。
* **`--dir_file`**: (类型: `str`, 默认: `''`)
    * **作用**: 如果提供此参数，脚本将加载这个预先计算好的HDF5文件作为投影方向，而不是重新计算PCA方向。这对于在相同的投影平面上比较不同轨迹非常有用。

**脚本内部如何使用这些参数：**

1.  **加载最终模型**: 使用 `--dataset`, `--model`, `--model_folder`, `--prefix`, `--max_epoch`, `--suffix` 来定位并加载训练到最后一个指定epoch的模型。这个模型将作为计算参数差异的基准。
2.  **收集模型文件列表**: 使用 `--model_folder`, `--prefix`, `--suffix`, `--start_epoch`, `--max_epoch`, `--save_epoch` 来构建一个包含所有相关训练阶段模型检查点文件路径的列表 (`model_files`)。
3.  **确定投影方向 (`dir_file`)**:
    * 如果 `--dir_file` 被指定，则直接使用该文件。
    * 否则，调用 `setup_PCA_directions` 函数。此函数会用到 `args` (包含了所有命令行参数，如 `--dataset`, `--model`, `model_files` (间接通过epoch参数得到), `--dir_type`, `--ignore`, `--save_epoch` 等) 以及最终模型的权重 `w` 和状态 `s` 来计算并保存PCA方向。
4.  **投影轨迹 (`project_trajectory`)**:
    * 这个函数需要 `dir_file` (包含投影方向), 最终模型的权重 `w` 和状态 `s`, 数据集名称 `args.dataset`, 模型名称 `args.model`, 模型文件列表 `model_files`, 方向类型 `args.dir_type`, 以及投影方法 (硬编码为 `'cos'`)。
5.  **绘图 (`plot_2D.plot_trajectory`)**:
    * 这个函数需要 `proj_file` (包含投影后坐标的HDF5文件) 和 `dir_file` (包含投影方向的HDF5文件，可能含有PCA方差解释比例等信息)。

因此，当你从命令行运行这个脚本时，你需要根据你的具体情况提供这些参数的值，以确保脚本能够正确地找到模型文件、计算或加载投影方向，并最终绘制出优化轨迹。



https://github.com/tomgoldstein/loss-landscape/issues/18
> 要在PCA（主成分分析）方向上绘制优化轨迹，您需要在训练期间准备好检查点文件。例如，您已经在CIFAR10数据集上训练了一个ResNet56模型300个周期 (epochs)，并且在 `path_to_model_folder` 路径下，每10个周期以 `model_{epoch}.pth` 的格式保存了检查点。绘制轨迹的命令如下：
>
> ```bash
> python plot_trajectory.py --dataset cifar10 --model resnet56 --model_folder path_to_model_folder \
> --start_epoch 0 --max_epoch 300 --save_epoch 10 --prefix model_ --suffix .pth
> ```
> （**注意**：原文中 `--surffix` 应该是一个笔误，根据代码上下文，它应该是 `--suffix`，我已在上面的命令中更正。如果您的脚本中确实是 `--surffix`，请以您的脚本为准。）
>
> 如果您想使用特定的方向文件而不是PCA方向，您可以使用选项 `--dir_file path_to_direction_file` 来指定。
>
> 请注意，由于以下原因，轨迹上的损失值与投影的损失等高线图并不完全对应：
>
> 1.  **损失景观的动态变化与BN层统计量的固定**：
>    * 损失景观在训练过程中是动态变化的，而我们通常只展示最终解附近的“局部”景观。为了绘制最终解附近的损失景观，我们必须固定最终的批量归一化（BN）统计量和参数。
>    * 然而，每个检查点的真实损失值应该根据训练过程中该检查点特定的BN统计量和参数来计算。
>    * 因此，那些远离最终解的检查点，其真实损失值与投影到最终解附近景观上的损失值之间会有较大的差异。
>    * 不过，当BN统计量稳定下来并接近最终值时，靠近最终解的轨迹上的投影损失值应该与真实值比较接近。
>
> 2.  **投影误差**：
>    * 通过两个PCA方向对轨迹进行投影是一种近似方法，在损失景观上恢复的对应点可能并不具有与检查点完全相同的模型参数。
>
> 3.  **PCA方向对优化末端的偏重**：
>    * 优化轨迹的末端通常聚集了更多的迭代点，而起始端的点较少。因此，由此产生的PCA方向会更精确地表示优化的末端，而不是优化过程的起始部分。
>
> 由于上述原因，准确捕捉SGD（随机梯度下降）的轨迹是很困难的。我们相信可以开发出更好的方法来绘制真实的SGD轨迹。


---
# 以下是原readme


This repository contains the PyTorch code for the paper
> Hao Li, Zheng Xu, Gavin Taylor, Christoph Studer and Tom Goldstein. [*Visualizing the Loss Landscape of Neural Nets*](https://arxiv.org/abs/1712.09913). NIPS, 2018.

An [interactive 3D visualizer](http://www.telesens.co/loss-landscape-viz/viewer.html) for loss surfaces has been provided by [telesens](http://www.telesens.co/2019/01/16/neural-network-loss-visualization/).

Given a network architecture and its pre-trained parameters, this tool calculates and visualizes the loss surface along random direction(s) near the optimal parameters.
The calculation can be done in parallel with multiple GPUs per node, and multiple nodes.
The random direction(s) and loss surface values are stored in HDF5 (`.h5`) files after they are produced.

## Setup

**Environment**: One or more multi-GPU node(s) with the following software/libraries installed:
- [PyTorch 0.4](https://pytorch.org/)
- [openmpi 3.1.2](https://www.open-mpi.org/)
- [mpi4py 2.0.0](https://mpi4py.scipy.org/docs/usrman/install.html)
- [numpy 1.15.1](https://docs.scipy.org/doc/numpy/user/quickstart.html)  
- [h5py 2.7.0](http://docs.h5py.org/en/stable/build.html#install)
- [matplotlib 2.0.2](https://matplotlib.org/users/installing.html)
- [scipy 0.19](https://www.scipy.org/install.html)

**Pre-trained models**:
The code accepts pre-trained PyTorch models for the CIFAR-10 dataset.
To load the pre-trained model correctly, the model file should contain `state_dict`, which is saved from the `state_dict()` method.
The default path for pre-trained networks is `cifar10/trained_nets`.
Some of the pre-trained models and plotted figures can be downloaded here:
- [VGG-9](https://drive.google.com/open?id=1jikD79HGbp6mN1qSGojsXOZEM5VAq3tH) (349 MB)
- [ResNet-56](https://drive.google.com/a/cs.umd.edu/file/d/12oxkvfaKcPyyHiOevVNTBzaQ1zAFlNPX/view?usp=sharing) (10 MB)
- [ResNet-56-noshort](https://drive.google.com/a/cs.umd.edu/file/d/1eUvYy3HaiCVHTzi3MHEZGgrGOPACLMkR/view?usp=sharing) (20 MB)
- [DenseNet-121](https://drive.google.com/a/cs.umd.edu/file/d/1oU0nDFv9CceYM4uW6RcOULYS-rnWxdVl/view?usp=sharing) (75 MB)

"ResNet-56-noshort" 中的 "noshort" 意味着这是一个 没有使用（或移除了）这种跳跃连接（shortcut connection）的ResNet-56变体

**Data preprocessing**:
The data pre-processing method used for visualization should be consistent with the one used for model training.
No data augmentation (random cropping or horizontal flipping) is used in calculating the loss values.

## Visualizing 1D loss curve

### Creating 1D linear interpolations
The 1D linear interpolation method [1] evaluates the loss values along the direction between two minimizers of the same network loss function. This method has been used to compare the flatness of minimizers trained with different batch sizes [2].
A 1D linear interpolation plot is produced using the `plot_surface.py` method.

```
mpirun -n 4 python plot_surface.py --mpi --cuda --model vgg9 --x=-0.5:1.5:401 --dir_type states \
--model_file cifar10/trained_nets/vgg9_sgd_lr=0.1_bs=128_wd=0.0_save_epoch=1/model_300.t7 \
--model_file2 cifar10/trained_nets/vgg9_sgd_lr=0.1_bs=8192_wd=0.0_save_epoch=1/model_300.t7 --plot
```
- `--x=-0.5:1.5:401` sets the range and resolution for the plot.  The x-coordinates in the plot will run from -0.5 to 1.5 (the minimizers are located at 0 and 1), and the loss value will be evaluated at 401 locations along this line.
- `--dir_type states` indicates the direction contains dimensions for all parameters as well as the statistics of the BN layers (`running_mean` and `running_var`). Note that ignoring `running_mean` and `running_var` cannot produce correct loss values when plotting two solutions togeather in the same figure.  
- The two model files contain network parameters describing the two distinct minimizers of the loss function.  The plot will interpolate between these two minima.

![VGG-9 SGD, WD=0](doc/images/vgg9_sgd_lr=0.1_bs=128_wd=0.0_save_epoch=1_model_300.t7_vgg9_sgd_lr=0.1_bs=8192_wd=0.0_save_epoch=1_model_300.t7_states.h5_[-1.0,1.0,401].h5_1d_loss_acc.jpg)



### Producing plots along random normalized directions
A random direction with the same dimension as the model parameters is created and "filter normalized."
Then we can sample loss values along this direction.

```
mpirun -n 4 python plot_surface.py --mpi --cuda --model vgg9 --x=-1:1:51 \
--model_file cifar10/trained_nets/vgg9_sgd_lr=0.1_bs=128_wd=0.0_save_epoch=1/model_300.t7 \
--dir_type weights --xnorm filter --xignore biasbn --plot
```
 - `--dir_type weights` indicates the direction has the same dimensions as the learned parameters, including bias and parameters in the BN layers.
 - `--xnorm filter` normalizes the random direction at the filter level. Here, a "filter" refers to the parameters that produce a single feature map.  For fully connected layers, a "filter" contains the weights that contribute to a single neuron.
 - `--xignore biasbn` ignores the direction corresponding to bias and BN parameters (fill the corresponding entries in the random vector with zeros).


 ![VGG-9 SGD, WD=0](doc/images/vgg9_sgd_lr=0.1_bs=128_wd=0.0_save_epoch=1/model_300.t7_weights_xignore=biasbn_xnorm=filter.h5_[-1.0,1.0,51].h5_1d_loss_acc.jpg)



We can also customize the appearance of the 1D plots by calling `plot_1D.py` once the surface file is available.


## Visualizing 2D loss contours

To plot the loss contours, we choose two random directions and normalize them in the same way as the 1D plotting.

```
mpirun -n 4 python plot_surface.py --mpi --cuda --model resnet56 --x=-1:1:51 --y=-1:1:51 \
--model_file cifar10/trained_nets/resnet56_sgd_lr=0.1_bs=128_wd=0.0005/model_300.t7 \
--dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn  --plot
```

![ResNet-56](doc/images/resnet56_sgd_lr=0.1_bs=128_wd=0.0005/model_300.t7_weights_xignore=biasbn_xnorm=filter_yignore=biasbn_ynorm=filter.h5_[-1.0,1.0,51]x[-1.0,1.0,51].h5_train_loss_2dcontour.jpg)

Once a surface is generated and stored in a `.h5` file, we can produce and customize a contour plot using the script `plot_2D.py`.

```
python plot_2D.py --surf_file path_to_surf_file --surf_name train_loss
```
- `--surf_name` specifies the type of surface. The default choice is `train_loss`,
- `--vmin` and `--vmax` sets the range of values to be plotted.
- `--vlevel` sets the step of the contours.


## Visualizing 3D loss surface
`plot_2D.py` can make a basic 3D loss surface plot with `matplotlib`.
If you want a more detailed rendering that uses lighting to display details, you can render the loss surface with [ParaView](http://paraview.org).

![ResNet-56-noshort](doc/images/resnet56_noshort_small.jpg) ![ResNet-56](doc/images/resnet56_small.jpg)

To do this, you must
1. Convert the surface `.h5` file to a `.vtp` file.
```
python h52vtp.py --surf_file path_to_surf_file --surf_name train_loss --zmax  10 --log
```
   This will generate a [VTK](https://www.kitware.com/products/books/VTKUsersGuide.pdf) file containing the loss surface with max value 10 in the log scale.

2. Open the `.vtp` file with ParaView. In ParaView, open the `.vtp` file with the VTK reader. Click the eye icon in the `Pipeline Browser` to make the figure show up. You can drag the surface around, and change the colors in the `Properties` window.

3. If the surface appears extremely skinny and needle-like, you may need to adjust the "transforming" parameters in the left control panel.  Enter numbers larger than 1 in the "scale" fields to widen the plot.

4. Select `Save screenshot` in the File menu to save the image.

## Reference

[1] Ian J Goodfellow, Oriol Vinyals, and Andrew M Saxe. Qualitatively characterizing neural network optimization problems. ICLR, 2015.

[2] Nitish Shirish Keskar, Dheevatsa Mudigere, Jorge Nocedal, Mikhail Smelyanskiy, and Ping Tak Peter Tang. On large-batch training for deep learning: Generalization gap and sharp minima. ICLR, 2017.

## Citation
If you find this code useful in your research, please cite:

```
@inproceedings{visualloss,
  title={Visualizing the Loss Landscape of Neural Nets},
  author={Li, Hao and Xu, Zheng and Taylor, Gavin and Studer, Christoph and Goldstein, Tom},
  booktitle={Neural Information Processing Systems},
  year={2018}
}
```
