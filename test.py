from model.waveunet import Waveunet
import museval
from tqdm import tqdm

import numpy as np
import torch

import data.utils
import model.utils as model_utils
import utils


def compute_model_output(model, inputs):
    '''
    Computes outputs of model with given inputs. Does NOT allow propagating gradients! See compute_loss for training.
    Procedure depends on whether we have one model for each source or not
    :param model: Model to train with
    :param compute_grad: Whether to compute gradients
    :return: Model outputs, Average loss over batch
    '''
    return model(inputs)


def predict(audio: torch.Tensor, model: Waveunet):
    '''
    Predict sources for a given audio input signal, with a given model. Audio is split into chunks to make predictions on each chunk before they are concatenated.
    :param audio: Audio input tensor, either Pytorch tensor or numpy array
    :param model: Pytorch model
    :return: Source predictions, dictionary with source names as keys
    '''
    if isinstance(audio, torch.Tensor):
        is_cuda = audio.is_cuda
        audio = audio.detach().cpu().numpy()
        return_mode = "pytorch"
    else:
        is_cuda = False
        return_mode = "numpy"

    expected_outputs = audio.shape[1]

    # Pad input if it is not divisible in length by the frame shift number
    output_shift = model.shapes["output_frames"]
    pad_back = audio.shape[1] % output_shift
    pad_back = 0 if pad_back == 0 else output_shift - pad_back
    if pad_back > 0:
        audio = np.pad(audio, [(0, 0), (0, pad_back)],
                       mode="constant", constant_values=0.0)

    target_outputs = audio.shape[1]
    output = np.zeros(audio.shape, np.float32)

    # Pad mixture across time at beginning and end so that neural network can make prediction at the beginning and end of signal
    pad_front_context = model.shapes["output_start_frame"]
    pad_back_context = model.shapes["input_frames"] - \
        model.shapes["output_end_frame"]
    audio = np.pad(audio, [(0, 0), (pad_front_context,
                   pad_back_context)], mode="constant", constant_values=0.0)

    # Iterate over mixture magnitudes, fetch network prediction
    with torch.no_grad():
        for target_start_pos in range(0, target_outputs, model.shapes["output_frames"]):
            # Prepare mixture excerpt by selecting time interval
            # Since audio was front-padded input of [targetpos:targetpos+inputframes] actually predicts [targetpos:targetpos+outputframes] target range
            curr_input = audio[:, target_start_pos:target_start_pos +
                               model.shapes["input_frames"]]

            # Convert to Pytorch tensor for model prediction
            curr_input = torch.from_numpy(curr_input).unsqueeze(0)

            # Predict
            output[:, target_start_pos:target_start_pos +
                   model.shapes["output_frames"]] = compute_model_output(model, curr_input).squeeze(0).cpu().numpy()

    # Crop to expected length (since we padded to handle the frame shift)
    outputs = output[:, :expected_outputs]

    if return_mode == "pytorch":
        outputs = torch.from_numpy(outputs)
        if is_cuda:
            outputs = outputs.cuda()
    return outputs


def predict_song(args, audio_path, model):
    '''
    Predicts sources for an audio file for which the file path is given, using a given model.
    Takes care of resampling the input audio to the models sampling rate and resampling predictions back to input sampling rate.
    :param args: Options dictionary
    :param audio_path: Path to mixture audio file
    :param model: Pytorch model
    :return: Source estimates given as dictionary with keys as source names
    '''
    model.eval()

    # Load mixture in original sampling rate
    audio, sr = data.utils.load(
        audio_path, sr=None, mono=False)  # type: ignore
    channels = audio.shape[0]
    len = audio.shape[1]

    # Adapt mixture channels to required input channels
    if args.channels == 1:
        audio = np.mean(audio, axis=0, keepdims=True)
    else:
        if channels == 1:  # Duplicate channels if input is mono but model is stereo
            audio = np.tile(audio, [args.channels, 1])
        else:
            assert(channels == args.channels)

    # resample to model sampling rate
    audio = data.utils.resample(audio, sr, args.sr)

    source = predict(audio, model)

    # Resample back to mixture sampling rate in case we had model on different sampling rate
    sources = data.utils.resample(source, args.sr, sr)

    # In case we had to pad the mixture at the end, or we have a few samples too many due to inconsistent down- and upsamṕling, remove those samples from source prediction now
    diff = source.shape[1] - len
    if diff > 0:
        print("WARNING: Cropping " + str(diff) + " samples")
        source = source[:, :-diff]
    elif diff < 0:
        print("WARNING: Padding output by " + str(diff) + " samples")
        source = np.pad(
            source, [(0, 0), (0, -diff)], "constant", 0.0)  # type: ignore

    # Adapt channels
    if channels > args.channels:
        assert(args.channels == 1)
        # Duplicate mono predictions
        source = np.tile(source, [channels, 1])
    elif channels < args.channels:
        assert(channels == 1)
        # Reduce model output to mono
        source = np.mean(source, axis=0, keepdims=True)

    # So librosa does not complain if we want to save it
    source = np.asfortranarray(source)

    return sources


def evaluate(args, dataset, model):
    '''
    Evaluates a given model on a given dataset
    :param args: Options dict
    :param dataset: Dataset object
    :param model: Pytorch model
    :return: Performance metric dictionary, list with each element describing one dataset sample's results
    '''
    perfs = list()
    model.eval()
    with torch.no_grad():
        for example in dataset:
            print("Evaluating " + example)

            # Load source references in their original sr and channel number
            target_source, _ = data.utils.load(
                example, sr=args.sr, mono=(args.channels == 1))

            # Predict using mixture
            pred_source = predict_song(args, example, model)
            # NOTE: わからん
            pred_source = np.stack([pred_source.T])  # type: ignore

            # Evaluate
            SDR, ISR, SIR, SAR, _ = museval.metrics.bss_eval(
                target_source, pred_source)

            song = {"SDR": SDR, "ISR": ISR,
                    "SIR": SIR, "SAR": SAR}
            perfs.append(song)

    return perfs


def validate(args, model, criterion, test_data):
    '''
    Iterate with a given model over a given test dataset and compute the desired loss
    :param args: Options dictionary
    :param model: Pytorch model
    :param criterion: Loss function to use (similar to Pytorch criterions)
    :param test_data: Test dataset (Pytorch dataset)
    :return:
    '''
    # PREPARE DATA
    dataloader = torch.utils.data.DataLoader(test_data,  # type: ignore
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.num_workers)

    # VALIDATE
    model.eval()
    total_loss = 0.
    with tqdm(total=len(test_data) // args.batch_size) as pbar, torch.no_grad():
        for example_num, (x, y) in enumerate(dataloader):
            if args.cuda:
                x = x.cuda()
                y = y.cuda()

            _, avg_loss = model_utils.compute_loss(
                model, x, y, criterion)

            total_loss += (1. / float(example_num + 1)) * \
                (avg_loss - total_loss)

            pbar.set_description("Current loss: {:.4f}".format(total_loss))
            pbar.update(1)

    return total_loss
