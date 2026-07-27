"""
dpu.py — shared FPGA / DPU helpers used by every AI session notebook.

The "DPU" (Deep-learning Processing Unit) is the part of the FPGA that runs the
AI model. Every AI notebook needs the same few steps:

    1. Load the overlay (the FPGA "app") and an AI model onto it.
    2. Set up input/output memory buffers the DPU reads from and writes to.
    3. Send an image in and get the model's raw numbers back.
    4. Turn those raw numbers into an easy answer (softmax + argmax).

That setup code is long and fiddly, so it lives here once and is shared by
PYNQ 101, 201, 301, 503, and any future AI session.
"""
import numpy as np
from pynq_dpu import DpuOverlay


def load_dpu(model_path, bitstream="dpu.bit"):
    """
    Load the FPGA overlay and an AI model onto the board.

    Parameters
    ----------
    model_path : str
        The model file to load, e.g. "dpu_resnet50.xmodel".
    bitstream : str
        The overlay ("app") for the FPGA. Almost always "dpu.bit".

    Returns
    -------
    overlay : DpuOverlay
        The loaded overlay. Use it for hardware (overlay.PMODA) and to
        pass into setup_dpu_buffers().

    Example
    -------
        overlay = load_dpu("dpu_mnist_classifier.xmodel")
    """
    overlay = DpuOverlay(bitstream)
    overlay.load_model(model_path)
    return overlay


def setup_dpu_buffers(overlay):
    """
    Wire up the DPU input/output memory buffers from a loaded overlay.

    The DPU needs empty NumPy arrays of exactly the right shape to read the
    image from and to write its answer into. Figuring out those shapes and
    making the arrays is the same in every notebook, so we do it here.

    Returns
    -------
    dpu : the DPU runner (used to actually run the model)
    input_data : list with one input array (put your image in input_data[0])
    output_data : list with one output array (the DPU fills this in)
    image_buf : shortcut to input_data[0] (the array your image goes into)
    shape_in : the shape the DPU expects the image to be
    shape_out : the shape of the answer the DPU gives back

    Example
    -------
        dpu, input_data, output_data, image, shape_in, shape_out = setup_dpu_buffers(overlay)
    """
    dpu = overlay.runner
    input_tensors = dpu.get_input_tensors()
    output_tensors = dpu.get_output_tensors()

    shape_in = tuple(input_tensors[0].dims)
    shape_out = tuple(output_tensors[0].dims)

    input_data = [np.empty(shape_in, dtype=np.float32, order="C")]
    output_data = [np.empty(shape_out, dtype=np.float32, order="C")]

    return dpu, input_data, output_data, input_data[0], shape_in, shape_out


def setup_dpu_buffers_multi(overlay):
    """
    Like setup_dpu_buffers, but for models that give BACK MORE THAN ONE output
    (for example the YOLO object-detection model in PYNQ 301, which has 3).

    Returns
    -------
    dpu, input_data, output_data, image_buf, shape_in, shape_outs

    where shape_outs is a tuple of output shapes (one per output tensor).

    Example
    -------
        dpu, input_data, output_data, image, shape_in, (s0, s1, s2) = setup_dpu_buffers_multi(overlay)
    """
    dpu = overlay.runner
    input_tensors = dpu.get_input_tensors()
    output_tensors = dpu.get_output_tensors()

    shape_in = tuple(input_tensors[0].dims)
    shape_outs = tuple(tuple(t.dims) for t in output_tensors)

    input_data = [np.empty(shape_in, dtype=np.float32, order="C")]
    output_data = [np.empty(s, dtype=np.float32, order="C") for s in shape_outs]

    return dpu, input_data, output_data, input_data[0], shape_in, shape_outs


def run_model(dpu, input_data, output_data, image=None):
    """
    Send one image through the DPU and get the raw model output back.

    Parameters
    ----------
    dpu : the DPU runner from setup_dpu_buffers()
    input_data, output_data : the buffer lists from setup_dpu_buffers()
    image : optional NumPy image already shaped like shape_in. If given, it is
            copied into the input buffer for you. If you already filled
            input_data[0] yourself, you can leave this as None.

    Returns
    -------
    output_data[0] : the raw output array the model produced.

    Example
    -------
        result = run_model(dpu, input_data, output_data, my_image)
    """
    if image is not None:
        input_data[0][0, ...] = image

    job_id = dpu.execute_async(input_data, output_data)
    dpu.wait(job_id)
    return output_data[0]


def softmax(values):
    """
    Turn a list of raw model scores into positive "confidence" numbers.

    Bigger raw score -> bigger confidence. This is the same simple version used
    in the MNIST notebook (exponentiate each value).

    Example
    -------
        confidences = softmax(raw_scores)
    """
    return np.exp(np.asarray(values).ravel())


def classify(output_data, output_size=None):
    """
    Turn a model output into a single answer: the class number it's most sure of.

    Parameters
    ----------
    output_data : the raw output from run_model() (or the full output list).
    output_size : optional number of classes. If not given, it is figured out
                  from the data automatically.

    Returns
    -------
    prediction : int   -> the class number with the highest confidence
    confidences : the full array of confidences (one per class)

    Example
    -------
        prediction, confidences = classify(result)
        print("The model thinks this is a", prediction)
    """
    # Accept either output_data[0] or the raw array itself.
    raw = output_data[0] if isinstance(output_data, (list, tuple)) else output_data
    flat = np.asarray(raw).ravel()
    if output_size is not None:
        flat = flat[:output_size]
    confidences = softmax(flat)
    prediction = int(confidences.argmax())
    return prediction, confidences
