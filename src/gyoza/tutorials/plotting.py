import matplotlib.pyplot as plt
import numpy as np 
from typing import Callable, List, Tuple
import tensorflow as tf
from matplotlib.backends.backend_agg import FigureCanvasAgg
from gyoza.modelling import flow_layers as gmfl
from gyoza.tutorials import data_synthesis as gtd

def __make_color_palette__() -> np.ndarray:
    """Generates interpolations between the colors red, green and blue.
    
    :return: color_palette (:class:`np.ndarray`) - An array of shape [55, 3], listing colors in RGB format."""

    # from https://github.com/JiahuiYu/generative_inpainting/blob/master/inpaint_ops.py
    RY, YG, GC, CB, BM, MR = (15, 6, 4, 11, 13, 6)
    ncols = RY + YG + GC + CB + BM + MR
    color_palette = np.zeros([ncols, 3])
    col = 0
    # RY
    color_palette[0:RY, 0] = 255
    color_palette[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY
    # YG
    color_palette[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    color_palette[col:col+YG, 1] = 255
    col += YG
    # GC
    color_palette[col:col+GC, 1] = 255
    color_palette[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC
    # CB
    color_palette[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    color_palette[col:col+CB, 2] = 255
    col += CB
    # BM
    color_palette[col:col+BM, 2] = 255
    color_palette[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM
    # MR
    color_palette[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    color_palette[col:col+MR, 0] = 255
    
    # Output
    return color_palette 

color_palette = __make_color_palette__()
"""A convenience variable, storing the color palette that is computed by :py:meth:`__make_color_palette__`."""

def make_2_dimensional_gaussian(mu: np.ndarray, sigma: np.ndarray, shape: List[int]) -> np.ndarray:
    """Generates a 2 dimensional Gaussian distribution.

    :param mu: The two means for the Gaussian variables. Assumed to be of shape [2].
    :type mu: np.ndarray
    :param sigma: The covariance matrix. Assumed to be of shape [2,2].
    :type sigma: np.ndarray
    :param shape: The desired shape of the output.
    :type shape: _type_, optional

    :return: 
        - X (:class:`numpy.ndarray`) - Coordiates of the grid of the two variables. Shape == [ ``shape`` [0]* ``shape`` [1],2].
        - p (:class:`numpy.ndarray`) - The probabilities associated with the coordinates ``X``. Shape == [ ``shape`` [0]* ``shape`` [1]].
        - D (:class:`numpy.ndarray`) - A matrix that arranges ``p`` with desired ``shape``.
        """


    # Generate x, y coordinates 
    x = np.linspace(-3, 3, shape[1])
    y = np.linspace(-3, 3, shape[0])
    xv, yv = np.meshgrid(x, y)
    X = np.concatenate([np.reshape(xv,[-1,1]), np.reshape(yv, [-1,1])], axis=1) # Shape == [shape[0]*shape[1],2] 
    
    # Compute probability
    numerator = np.exp(-0.5*np.sum((X-mu).dot(np.linalg.inv(sigma)) * (X-mu), axis=1))
    denominator = np.sqrt((2*np.pi)**2 * np.linalg.det(sigma))
    p = numerator / denominator

    D = np.reshape(p, [shape[1], shape[0]])

    # Outputs
    return X, p, D

def swirl(x:np.ndarray, y: np.ndarray, x0:float = 0, y0: float = 0, radius: float = 5, rotation: float = 0, strength: float = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Performs a swirl operation on given x and y coordinates.
    
    Inputs:
    - x, y: Coordinates of points that shall be swirled.
    - x0, y0: The origin of the swirl.
    - radius: The extent of the swirl. Small values indicate local swirl, large values indicate global swirl.
    - rotation: Adds a rotation angle to the swirl.
    - strength: Indicates the strength of swirl.

    Outputs:
    - x_new, y_new: The transformed coordinates.
    """
    
    # Polar coordinates of each point
    theta = np.arctan2((y-y0), (x-x0))
    rho = np.sqrt((x-x0)**2 + (y-y0)**2)
    
    # Swirl
    r = np.log(2)*radius/5
    new_theta = rotation + strength * np.exp(-rho/r) + theta

    # Cartesian coordinates
    x_new = rho * np.cos(new_theta)
    y_new = rho * np.sin(new_theta)

    # Outputs
    return x_new, y_new

def make_radial_line(radius: float, rotation: float, point_count: int) -> np.ndarray:
    """Generates a straight line with ``point_count`` many points that has one endpoint at the origin and the other endpoint on the 
    circle defined by defined by ``radius`` and ``rotation``.

    :param radius: The radius of the circle from which lines are generated.
    :type radius: float
    :param rotation: The angle of rotation of the line in radians. Movement is clockwise.
    :type rotation: float
    :param point_count: The number of points on the line.
    :type point_count: int
    :return: x, y (:class:`np.ndarray`) - The coordinates of line with shape [``point_count``, 2].
    """

    # Generate horizontal line
    x = np.arange(start=0, stop=radius+radius/point_count, step=radius/point_count, dtype=np.float32)
    y = np.zeros(x.shape, dtype=np.float32)
    line = np.concatenate([x[:,np.newaxis], y[:,np.newaxis]], axis=1); del x, y

    # Rotate
    rotaton_matrix = np.array([[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]])
    line = np.dot(line, rotaton_matrix)

    # Unpack
    x = line[:,0]; y = line[:,1]

    # Outputs
    return x, y

def make_color_wheel(pixels_per_inch: int, pixel_count: int = 128, swirl_strength: float = 0, gaussian_variance: float = 1) -> np.ndarray:
    """Generates an image of a color wheel with swirl

    :param dpi: The density of pixels per inch on the user machine.
    :type dpi: int
    :param pixel_count: The desired width and height of ``image`` in pixels, defaults to 128
    :type pixel_count: int, optional
    :param swirl_strength: The strength of swirl applied to the color wheel. Sensible values are in the range [0,10]. The sign is 
        ignored. Defaults to 0
    :type swirl_strength: float, optional
    :param saturation: The saturation of the colors. valid values are in range [0,1], where 0 corresponds to a white image and 1 to
        a fully satured image. Defaults to 1.
    :type saturation: float, optional
    :return: image (:class:`np.ndarray`) - The image of shape [pixel_count, pixel_count, 4] where 4 are the channels.
    """

    # Make radial lines
    x_s = [None] * len(color_palette); y_s = [None] * len(color_palette)
    for c in range(len(color_palette)):
        # Make straight line
        x, y = make_radial_line(radius=1, rotation=c*2*np.pi/(len(color_palette)), point_count=2+(int)(2*swirl_strength))
        
        # Add swirl
        if swirl_strength != 0: x,y = swirl(x=x,y=y,radius=5, rotation=0, strength=swirl_strength)
            
        # Save to array
        x_s[c] = x; y_s[c] = y

    # Draw wedges
    figure = plt.figure(figsize=(pixel_count/(2*pixels_per_inch), pixel_count/(2*pixels_per_inch)), dpi=pixels_per_inch)
    plt.axis('off')
    for c, color in enumerate(color_palette):
        # A wedges has two main lines
        x_c = x_s[c]; y_c = y_s[c]
        x_d = x_s[(c+1) % len(color_palette)]; y_d = y_s[(c+1) % len(color_palette)]
        
        # Draw on figure
        plt.fill(np.concatenate([x_c,np.flip(x_d)]), np.concatenate([y_c,np.flip(y_d)]), color=tuple(color/255), linewidth=1/pixels_per_inch)

    plt.xlim([-1,1]); plt.ylim([-1,1])
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
    # Export as image
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    s, (width, height) = canvas.print_to_buffer()
    image = np.fromstring(s, np.uint8).reshape((height, width, 4))
    plt.close()

    # Apply gaussian saturation
    _, _, gaussian = make_2_dimensional_gaussian(mu=np.zeros([2]), sigma=gaussian_variance*np.eye(2), shape=[height, width])
    gaussian = (gaussian-np.min(gaussian))/np.max(gaussian) # Now ranges between 0 and 1
    gaussian = np.array(255 *gaussian, dtype=np.uint8)
    image[:,:,3] = gaussian # The alpha channel 
    
    # Outputs
    return image

def plot_instance_pairs(S: np.ndarray, Z_a: np.ndarray, Z_b: np.ndarray, Y_ab: np.ndarray, manifold_function: Callable, manifold_name: str, pair_count: int=3):
    """Plots pairs of instances along with their similarities and the manifold (without noise).

    :param S: The position along the manifold. Shape == [:math:`M`, 1], where :math:`M` is the number of instances.
    :type S: np.array
    :param Z_a: The coordinates of the a-instances to be plotted. Shape is assumed to be [:math:`M`, :math:`N`], where :math:`M` is the number
        of instances and :math:`N = 2` is the dimensionality of an instance.
    :type Z_a: :class:`numpy.ndarray`
    :param Z_b: The same as Z_a, but for b-instances.
    :type Z_b: :class:`numpy.ndarray`
    :param Y_ab: The similarities of the ``Z_ab`` instances. Shape is assumed to be [:math:`M`, :math:`F`], where :math:`M` is the number of
        instances and :math:`F=2` at axis 1 is the factor count.
    :type Y_ab: :class:`numpy.ndarray`
    :param manifold_function: A function that takes as input the position ``S`` along the manifold and provides as output the two coordinates
        that are associated with that position along the manifold. Hence, [:math:`M`,1] -> [:math:`M`, :math:`N`], where M is the number of
        instances and :math:`N=2` their dimensionality
    :type manifold_function: Callable
    :param manifold_name: A name assigned to the manifold that is used as a label in the plot.
    :type manifold_name: str
    :param pair_count: The number of pairs to be illustrated
    :type pair_count: int, optional, defaults to 3
    """

    # Construct figure
    plt.figure(figsize=(3.5,3.5)); plt.title(rf"Noisy Instances, Pairs and Similarities (s) for ${manifold_name}$")

    # Plot instance pairs
    individual_Z = np.concatenate([Z_a, Z_b], axis=1)
    plt.scatter(individual_Z[:,0], individual_Z[:,1], c='lightgray')
    plt.scatter(Z_a[:pair_count,0], Z_a[:pair_count,1]) # Instances a
    plt.scatter(Z_b[:pair_count,0], Z_b[:pair_count,1]) # Instances b
    
    # Plot manifold
    plt.scatter(*manifold_function(S), color='gray', marker='.', s=1)

    # Label current plots
    plt.legend(['z','z_a','z_b',rf"${manifold_name}$"])

    # Plot lines that connect the two instances of a pair
    s_max = np.max(S)
    for i in range(pair_count): # Iterate instances
        plt.plot([Z_a[i,0], Z_b[i,0]], [Z_a[i,1], Z_b[i,1]], '--', color='black')
        plt.text(Z_b[i,0]+0.1, Z_b[i,1], 's = ' + str(np.round(Y_ab[i,1], 3))) # Label for their similarity
    plt.gca().set_aspect('equal')
    #plt.ylim(-s_max/0.8,s_max/0.8); plt.xlim(-s_max/0.8,s_max/0.8)

    plt.xlabel('First Dimension'); plt.ylabel('Second Dimension')
    plt.show()

def plot_instance_pairs_2(Z_a: np.ndarray, Z_b: np.ndarray, title_suffix: str = rf"$Z$"):
    """Plots the instance pairs of Z_ab (or Z_tilde_ab) in two scatter plots. The first scatter plot shows the first dimension (index 0) of instance a and b while the second scatter plot shows the second dimension (index 1) of instances a and b.
    In the margins of each scatter plot, the marginal histograms are shown.
    
    :param Z_a: The coordinates of the a-instances to be plotted. Shape is assumed to be [:math:`M`, :math:`N`], where :math:`M` is the number
        of instances and :math:`N = 2` is the dimensionality of an instance.
    :type Z_a: :class:`numpy.ndarray`
    :param Z_b: The same as Z_a, but for b-instances.
    :type Z_b: :class:`numpy.ndarray`
    :param title_suffix: The suffix to be added to the title, usually a string 'Z' to indicate that instances come from the Z-space or rf'$\tilde{Z}$' to indicate that they come from the Z_tilde-space.
    :type title_suffix: str, optional, defaults to rf'$Z$'
    """

    fig, axs = plt.subplots(2,4,figsize=(9,4.5), gridspec_kw={'height_ratios': [4, 0.5], 'width_ratios':[0.5,4,0.5,4]})

    plt.suptitle("Instance Pairs " + title_suffix)

    # Iterate dimensions
    for d in range(2):
        plt.subplot(2,4,d*2+2)
        plt.title(("First" if d==0 else "Second") + " Dimension ")
        plt.scatter(Z_a[:,d], Z_b[:,d], s=0.5, c='k')
        plt.xlabel(f"Instance a"); plt.ylabel(f"Instance b")
        Z_a_0_lim, Z_b_0_lim = plt.xlim(), plt.ylim()
        plt.legend([f"r = {np.round(np.corrcoef(Z_a[:,d], Z_b[:,d])[0,1], 3)}"])

        # Histograms
        plt.subplot(2,4,d*2+6)
        plt.hist(Z_a[:,d], histtype='step', color='k'); plt.xlim(Z_a_0_lim); plt.gca().invert_yaxis(); plt.axis('off')
        plt.subplot(2,4,d*2+1)
        plt.hist(Z_b[:,d], orientation='horizontal', histtype='step', color='k'); plt.ylim(Z_b_0_lim); plt.gca().invert_xaxis(); plt.axis('off')

        # Disable corner subplot
        plt.subplot(2,4,d*2+5); plt.axis('off')

    plt.tight_layout()
    plt.show()

def plot_loss_trajectory(epoch_loss_means: List[float], epoch_loss_standard_deviations: List[float], manifold_name: str):
    """Plots the loss trajectory after model calibration with error surface.

    :param epoch_loss_means: The mean across batches for each epoch. Length = [epoch count]
    :type epoch_loss_means: List[float]
    :param epoch_loss_standard_deviations:  The standard deviation across batches for each epoch. Length = [epoch count]
    :type epoch_loss_standard_deviations: List[float]
    :param manifold_name: The name of the manifold on which the model was calibrated. Used for the title.
    :type manifold_name: str
    """

    # Preprocess
    M = len(epoch_loss_means)
    means = np.array(epoch_loss_means)
    errors = 2.0 * np.array(epoch_loss_standard_deviations) / np.sqrt(M)

    # Create figure
    plt.figure(figsize=[12,3]); plt.title(rf'Loss Trajectory on Manifold ${manifold_name}$')

    # Error surface
    plt.fill_between(x=list(range(M)), y1=means-errors, y2=means+errors, color='mistyrose')

    # Line
    plt.plot(epoch_loss_means)

    # Labels
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend([r'$\pm 2*$ Standard Error', 'Mean Across Batches'])

def plot_input_output(network: gmfl.FlowModel, S, manifold_function: Callable, noise_standard_deviation: float, manifold_name: str, zoom_output: bool=False):
    """Plots the input and output to the ``network``. Points are colored using a color wheel. Supplementary marginal distribution are provided.

    :param network: The network that shall process the data. It is expected to map from [:math:`M`,:math:`N`] to [:math:`M`,:math:`N`], where
        :math:`M` is the instance count and :math:`N=2` the dimensionality.
    :type network: mfl.SupervisedFactorNetwork
    :param S: A one-dimensional array providing the position along the manifold.
    :type S_range: numpy.array
    :param manifold_function: A function that maps from position on manifold (:math:`S`, shape == [instane count M]) to coordinates in :math:`N=2`
        dimensional space.
    :type manifold_function: Callable
    :param noise_standard_deviation: Standard deviation of the noise that shall be added to the data before passing it through the model.
    :type noise_standard_deviation: float
    :param manifold_name: The name of the manifold used in the title.
    :type manifold_name: str
    :param zoom_output: Indicates whether the output should have the same zoom as the input (False) or should be zoomed according to its own scale (True).
    :type zoom_output: bool, optional, defaults to False
    """

    # Sample from manifold to illustrate distortion of data
    S = np.random.choice(a=S, size=len(color_palette), replace=False)#np.arange(np.min(S), np.max(S), (np.max(S) - np.min(S))/len(color_palette)) # Each point will receive its own color
    S = np.sort(S) # Ensure colors are ascending
    Z, Y = gtd.create_data_set(S=S, manifold_function=manifold_function, noise_standard_deviation=noise_standard_deviation)
    x_1, x_2 = Z[:,0], Z[:,1]

    # Create gridlines to illustrate distortion of surrounding space
    points_per_line = 10
    min_x_1 = np.min(x_1); max_x_1 = np.max(x_1); mean_x_1 = np.abs(np.mean(x_1)) # Horizontal ticks
    min_x_2 = np.min(x_2); max_x_2 = np.max(x_2); mean_x_2 = np.abs(np.mean(x_2)) # Vertical ticks

    if (np.abs(max_x_1 - min_x_1) < np.abs(max_x_2 - min_x_2)): # Ensures lines are large enough to encompass data and square
        x_2_grid = np.linspace(min_x_2 - np.abs(mean_x_2 - min_x_2), max_x_2 + np.abs(mean_x_2-max_x_2), points_per_line)
        h_x_1, h_x_2 = np.meshgrid(x_2_grid, x_2_grid) # horizontal line coordinates
        v_x_2, v_x_1 = np.meshgrid(x_2_grid, x_2_grid) # vertical line coordinates
    else:
        x_1_grid = np.linspace(min_x_1 - np.abs(mean_x_1-min_x_1), max_x_1 + np.abs(mean_x_1-max_x_1), points_per_line)
        h_x_1, h_x_2 = np.meshgrid(x_1_grid, x_1_grid) # horizontal line coordinates
        v_x_2, v_x_1 = np.meshgrid(x_1_grid, x_1_grid) # vertical line coordinates

    # Plot
    fig, axs = plt.subplots(2,4,figsize=(9,4.5), gridspec_kw={'height_ratios': [4, 0.5], 'width_ratios':[0.5,4,0.5,4]})

    # 1. Plot joint distributions
    # 1.1 Z
    plt.suptitle(rf"Inference on ${manifold_name}$")
    plt.subplot(2,4,2); plt.title("Input")

    # 1.1.1 Gridlines
    for l in range(points_per_line): plt.plot(h_x_1[l,:], h_x_2[l,:], color='#C5C9C7', linewidth=0.75)
    for l in range(points_per_line): plt.plot(v_x_1[l,:], v_x_2[l,:], color='#C5C9C7', linewidth=0.75)

    # 1.1.2 Data
    Z = np.concatenate([x_1[:,np.newaxis], x_2[:, np.newaxis]], axis=1)
    plt.scatter(Z[:,0], Z[:,1], c=color_palette/255.0, zorder=3); plt.xlabel("First Dimension"); plt.ylabel("Second Dimension")
    Z_x_lim = plt.gca().get_xlim(); Z_y_lim = plt.gca().get_ylim() # Use these for marginal distributions

    # 1.2 Z tilde
    plt.subplot(2,4,4); plt.title("Output")
    Z_tilde, _ = network(Z)
    H_Z_tilde, _ = network(tf.concat([np.reshape(h_x_1, [-1])[:,np.newaxis], np.reshape(h_x_2, [-1])[:,np.newaxis]], axis=1))
    V_Z_tilde, _ = network(tf.concat([np.reshape(v_x_1, [-1])[:,np.newaxis], np.reshape(v_x_2, [-1])[:,np.newaxis]], axis=1))
    
    # 1.2.1 Gridlines
    for l in range(points_per_line): plt.plot(H_Z_tilde[l*points_per_line:(l+1)*points_per_line,0], H_Z_tilde[l*points_per_line:(l+1)*points_per_line,1], color='#C5C9C7', linewidth=0.75)
    for l in range(points_per_line): plt.plot(V_Z_tilde[l*points_per_line:(l+1)*points_per_line,0], V_Z_tilde[l*points_per_line:(l+1)*points_per_line,1], color='#C5C9C7', linewidth=0.75)

    # 1.2.2 Data
    plt.scatter(Z_tilde[:,0], Z_tilde[:,1], c=color_palette/255.0, zorder=3); plt.xlabel('Residual Factor'); plt.ylabel('Manifold Position Factor')
    Z_tilde_min = np.min(Z_tilde, axis=0); Z_tilde_max = np.max(Z_tilde, axis=0); Z_tilde_mean = np.mean(Z_tilde, axis=0)
    Z_tilde_x_lim = Z_x_lim if not zoom_output else (Z_tilde_min[0] - np.abs(Z_tilde_mean[0] - Z_tilde_min[0]), Z_tilde_max[0] + np.abs(Z_tilde_mean[0] - Z_tilde_max[0]))
    Z_tilde_y_lim = Z_y_lim if not zoom_output else (Z_tilde_min[1] - np.abs(Z_tilde_mean[1] - Z_tilde_min[1]), Z_tilde_max[1] + np.abs(Z_tilde_mean[1] - Z_tilde_max[1]))
    plt.xlim(*Z_tilde_x_lim); plt.ylim(*Z_tilde_y_lim)

    # 2. Plot marginal distributions
    # 2.1 Z
    plt.subplot(2,4,6)
    plt.hist(Z[:,0], histtype='step'); plt.xlim(Z_x_lim); plt.gca().invert_yaxis(); plt.axis('off')
    plt.subplot(2,4,1)
    plt.hist(Z[:,1], orientation='horizontal', histtype='step'); plt.ylim(Z_y_lim); plt.gca().invert_xaxis(); plt.axis('off')

    # 2.2 Z tilde
    plt.subplot(2,4,8)
    plt.hist(Z_tilde[:,0], histtype='step'); plt.gca().invert_yaxis(); plt.xlim(Z_tilde_x_lim); plt.axis('off')
    plt.subplot(2,4,3)
    plt.hist(Z_tilde[:,1], orientation='horizontal', histtype='step'); plt.ylim(Z_tilde_y_lim); plt.gca().invert_xaxis(); plt.axis('off')

    # Make other subplots invisible
    plt.subplot(2,4,5); plt.axis('off')
    plt.subplot(2,4,7); plt.axis('off')

    plt.tight_layout()
    plt.show()

def evaluate_and_plot_networks(Z_test: List[np.ndarray], Y_test: List[np.ndarray], networks: List[gmfl.FlowModel], manifold_name: str):
    """For each network, a scatter plot for the predicted and actual position along the manifold is plotted along with a bar for the proportion
    of explained variance.

    :param Z_test: A list of test sets used as input to the corresponding network in ``networks``. The list is expected to have the same length
        as ``networks`` and each test set is assumed to have shape [:math:`M^*`,:math:`N`], where :math:`M^*` is the number of instances in a
        test set and :math:`N=2` is the dimensinoality of an instance.
    :type Z_test: List[np.ndarray]
    :param Y_test: A list of test sets used to evaluate to the corresponding network in ``netwroks``. The list is expected to have the same length
        as ``networks`` and each test set is assumed to have shape [:math:`M^*`,:math:`F`], where :math:`M^*` is the number of instances in a
        test set and :math:`F=2` is the number of factors. It is assumed that factor at index 1 encodes the position along the manifold.
    :type Y_test: List[np.ndarray]
    :param networks: A list of calibrated networks that take ``Z_test`` as input and whose output (of same shape as input) encodes position along
        the data manifold along index 2.
    :type networks: List[mfl.SupervisedFactorNetwork]
    :param manifold_name: The name of the manifold used in the figure title.
    :type manifold_name: str
    """

    # Prepare figure
    fold_count = len(networks)
    fig, axs = plt.subplots(1,3*fold_count,figsize=(12,2), gridspec_kw={'width_ratios': [6, 1, 1]*fold_count})
    plt.suptitle(rf"{fold_count}-Fold Cross Validated Evaluation on ${manifold_name}$")

    # Iterate networks
    for n, network in enumerate(networks):
        # 1. Predict
        Y_hat = network(Z_test[n]).numpy()
        Y = Y_test[n]

        # 2. Create scatter plot for manifold position
        plt.subplot(1,3*fold_count, 3*n+1)
        plt.scatter(Y_hat[:,1], Y[:,1], color='black', marker='.', s=1)
        ax = plt.gca();ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

        if n == 0:
            plt.ylabel("Manifold Position\n\nActual"); plt.xlabel("Predicted")
        else:
            plt.yticks([]); plt.xlabel(f'Fold {n+1}')

        # 3. Create bar plot for proportion of explained variance for position along manifold (factor at position 1)
        r2 = np.corrcoef(Y[:,1], Y_hat[:,1])[1,0]**2 # Since the output is a correlation matrix, the [1,0] selects the correlation between the two variables

        plt.subplot(1,3*fold_count, 3*n+2);
        plt.bar([0], [r2], color='white', edgecolor='black'); plt.ylim([0,1]); plt.xlim([-1,1]); plt.xticks([])#[n+1], [f"Fold {n+1}"])
        if n==0:
            ax = plt.gca();ax.spines['top'].set_visible(False); ax.spines['left'].set_visible(False); ax.spines['right'].set_visible(False)
            plt.ylabel(r'$r^2$');  plt.gca().yaxis.tick_right();  plt.yticks([0,1])
        else:plt.axis('off')

        t=plt.text(-0,r2/2,f'{np.round(r2,2)}', horizontalalignment='center'); t.set_bbox(dict(facecolor='white', alpha=0.9, edgecolor='white'))

        # Padding on right of each bar plot
        plt.subplot(1,3*fold_count, 3*n+3); plt.axis('off')

    plt.show()

def plot_inverse_point(position: float, residual: float, S: np.ndarray, network: gmfl.FlowModel, manifold_function: Callable, manifold_name: str):
    """This function visualizes the ``network``'s inversion ability by plotting the inverse of the point[``residual``, ``position``]. It also
    plots the ``manifold_function`` on input ``S`` for reference.

    :param position: The position along the manifold that shall be entered for dimensions at index 1 for inversion via the ``network``.
    :type position: float
    :param residual: The residual that shall be entered for dimensions at index 0 for inversion via the ``network``.
    :type residual: float
    :param S: Points along which the ``manifold_function`` shall be evaluated during plotting.
    :type S: :class:`numpy.ndarray`
    :param network: A network calibrated to disentangle manifold position (factor at dimension 1) from deviation from manifold (factor at
        dimension 0). It shall map from [:math:`M`,:math:`N`] to [:math:`M`,:math:`N`], where :math:`M` is the instance count and :math:`N=2`
        is the dimensionality.
    :type network: mfl.SupervisedFactorNetwork
    :param manifold_function: A function that maps from position on manifold (:math:`S`, shape == [instane count M]) to coordinates in :math:`N=2`
        dimensional space.
    :type manifold_function: _type_
    :param manifold_name: The name of the manifold used for the figure title.
    :type manifold_name: str
    """

    # Construct figure
    plt.figure(figsize=(3.5,3.5)); plt.title(rf"Inverse Modelling on ${manifold_name}$")

    # Predict position using network
    Z_tilde = tf.constant([[residual,position]], dtype=tf.keras.backend.floatx())
    Z = network.invert(Z_tilde)
    plt.scatter(Z[:,0], Z[:,1])

    # Plot manifold
    plt.scatter(*manifold_function(S), color='gray', marker='.', s=1)

    # Axes and labels
    s_max = np.max(S)
    plt.ylim(-s_max/0.8,s_max/0.8); plt.xlim(-s_max/0.8,s_max/0.8)
    plt.xlabel('First Dimension'); plt.ylabel('Second Dimension')
    plt.show()

def plot_contribution_per_layer(network: gmfl.FlowModel, s_range: Tuple[float, float], manifold_function: Callable, manifold_name:str, layer_steps: List[int], step_titles: List[str]):
    """Plots for each layer (or rather step of consecutive layers) the contribution to the data transformation. The plot is strucutred into three rows.
    The first row shows a stacked bar chart whose bottom segment is the contribution due to affine transformation and the top segment is the contribution
    due to higher order transformation. To better understand the mechanisms behind these contributions there is a pictogram in the bottom row for the
    actual affine transformation and in the middle row for the remaining higher order part. This separation is done to understand the complexity of the
    transformation, whereby affine is considered simple and higher order is considered complex. The decomposition into affine and higher order is obtained
    by means of a first order `Maclaurin series <https://en.wikipedia.org/wiki/Taylor_series#Taylor_series_in_several_variables>`_.

    :param network: The network whose transfromation shall be visualized. It is expecetd to map 1 dimensional manifolds from the real 2-dimensional
      plane to the real 2-dimensional plane.
    :type network: :class:`gyoza.modelling.flow_layers.FlowModel`
    :param s_range: The lower and upper bounds for the position along the manifold, respectively.
    :type s_range: Tuple[float, float]
    :param manifold_function: A function that maps from position along manifold to coordinates on the manifold in the real two dimensional plane.
    :type manifold_function: :class:`Callable`
    :param manifold_name: The name of the manifold used for the figure title.
    :type manifold_name: str
    :param layer_steps: A list of steps across layers of the ``network``. If, for instance, the network has 7 layers and visualization shall be done for
      after the 1., 3. and 7, then ``layer_steps`` shall be set to [1,3,7]. The minimum entry shall be 1, then maximum entry shall be the number of layers
      in ``network`` and all entries shall be strictly increasing.
    :type layer_steps: List[int]
    :param step_titles: The titles associated with each step in ``layer_steps``. Used as titles in the figure.
    :type step_titles: List[str]
    """

    # Prepare plot
    #plt.figure(figsize=(12,3.5));
    layer_steps = [0] + layer_steps
    K = len(step_titles)
    fig, axs = plt.subplots(3, 1+K, figsize=(0.8+K,5), gridspec_kw={'height_ratios': [2,1,1], 'width_ratios':[0.3]+[1]*K})
    plt.suptitle(rf'Contribution per Layer on ${manifold_name}$')

    # Sample from s range
    S = np.linspace(s_range[0], s_range[1], len(color_palette), dtype=tf.keras.backend.floatx())
    z_1, z_2 = manifold_function(S); Z = np.concatenate([z_1[:, np.newaxis], z_2[:, np.newaxis]], axis=1)
    max_bar_height = 0

    # Plot annotations on left
    gray = [0.8,0.8,0.8]
    #plt.subplot(3,1+K,1); plt.axis('off')
    plt.subplot(3,1+K,1+K+1); plt.bar([''],[1], color=gray, edgecolor='black', hatch='oo'); plt.ylim(0,1); plt.xticks([]); plt.yticks([]); plt.ylabel('Higher Order')
    plt.subplot(3,1+K,2*(1+K)+1); plt.bar([''],[1], color=gray, edgecolor='black', hatch='///'); plt.ylim(0,1); plt.xticks([]); plt.yticks([]); plt.ylabel('Affine')

    # Iterate layers
    for k in range(1, len(layer_steps)):

        # Set up 1st order Maclaurin decomposition https://en.wikipedia.org/wiki/Taylor_series#Taylor_series_in_several_variables
        # Z_tilde ~= layer(0) + J(0) * Z, where J(0) is the jacobian w.r.t input evaluated at the origin
        origin = tf.Variable(tf.zeros([1] + list(Z[0].shape), dtype=tf.keras.backend.floatx())) # The extra 1 is the batch dimension
        Z_tilde = Z
        c = origin # Shape == [1, N]. The layer's shifting of the origin
        with tf.GradientTape() as tape:
          for layer in network.layers[layer_steps[k-1]:layer_steps[k]]:
            c = layer(c)
            Z_tilde = layer(Z_tilde) # Shape == [instance count, N]

        J = tf.squeeze(tape.jacobian(c, origin)) # Shape == [N z_tilde dimensions, N z dimensions]. The layer's linear combination of input dimensions

        # Compute approximation error (contribution of higher order terms in the Maclaurin series)
        prediction = c + tf.linalg.matmul(Z, tf.transpose(J))
        P = prediction - Z # Shape == [instance count, N]. Arrows from Z to prediction
        E = Z_tilde - prediction # Shape == [instance count, N]. Arrows from prediction to Z_tilde

        # 2. Plot
        # 2.1 Bars
        plt.subplot(3,1+K,k+1); plt.title(step_titles[k-1], fontsize=10)
        E_norm = np.mean(np.sqrt(np.sum(E**2, axis=1)))
        P_norm = np.mean(np.sqrt(np.sum(P**2, axis=1)))
        plt.bar([''],[E_norm+P_norm], color = gray, edgecolor='black', hatch='oo')
        plt.bar([''],[P_norm], color = gray, edgecolor='black', hatch='///')
        max_bar_height = max(max_bar_height, E_norm+P_norm); plt.axis('off')

        # 2.1 Tails
        # 2.1.1 Error
        plt.subplot(3,1+K,1+K+k+1)
        plt.scatter(prediction[:,0], prediction[:,1], color=gray, marker='.',s=0.1)
        plt.quiver(prediction[:,0], prediction[:,1], E[:,0], E[:,1], angles='xy', scale_units='xy', scale=1., color=gray, zorder=3)
        plt.scatter(Z_tilde[:,0], Z_tilde[:,1], c=color_palette/255.0, marker='.',s=1.5)
        plt.axis('equal'); plt.xticks([]); plt.yticks([]); plt.xlim(1.3*np.array(plt.xlim())); plt.ylim(1.3*np.array(plt.ylim()))

        # 2.1.2 Prediction
        plt.subplot(3,1+K,2*(1+K)+k+1)
        plt.scatter(Z[:,0], Z[:,1], color=gray, marker='.',s=0.1)
        plt.quiver(Z[:,0], Z[:,1], P[:,0], P[:,1], angles='xy', scale_units='xy', scale=1., color=gray, zorder=3)
        plt.scatter(prediction[:,0], prediction[:,1], c=color_palette/255.0, marker='.',s=1.5)
        plt.axis('equal'); plt.xticks([]); plt.yticks([]); plt.xlim(1.3*np.array(plt.xlim())); plt.ylim(1.3*np.array(plt.ylim()))

        # Prepare next iteration
        Z=Z_tilde

    # Adjust bar heights
    for k in range(1, len(layer_steps)):
      plt.subplot(3,1+K,k+1); plt.ylim(0, max_bar_height)
    plt.subplot(3,1+K,1); plt.ylabel('Mean Change'); plt.ylim(0, max_bar_height); ax = plt.gca();ax.spines['top'].set_visible(False); ax.spines['left'].set_visible(False); ax.spines['bottom'].set_visible(False); plt.xticks([])
    ax.yaxis.tick_right(); ax.tick_params(axis="y",direction="in", pad=-12)

    plt.tight_layout()
    plt.show()