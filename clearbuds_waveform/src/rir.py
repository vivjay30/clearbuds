import numpy as np
import librosa
import pyroomacoustics as pra
import random

def generate_mic_array(room, mic_radius: float, n_mics: int):
    """
    Generate a list of Microphone objects
    Radius = 50th percentile of men Bitragion breadth
    (https://en.wikipedia.org/wiki/Human_head)
    """
    R = pra.circular_2D_array(center=[0., 0.], M=n_mics, phi0=0, radius=mic_radius)
    room.add_microphone_array(pra.MicrophoneArray(R, room.fs))


def rir_perturbation(audio1, audio2, sample_rate):
    # Generate room parameters, each scene has a random room and absorption
    assert(audio1.shape == audio2.shape)
    total_samples = audio1.shape[0]

    # left_wall = np.random.uniform(low=-20, high=-15)
    # right_wall = np.random.uniform(low=15, high=20)
    # top_wall = np.random.uniform(low=15, high=20)
    # bottom_wall = np.random.uniform(low=-20, high=-15)
    # absorption = np.random.uniform(low=0.1, high=0.2)
    # corners = np.array([[left_wall, bottom_wall], [left_wall, top_wall],
    #                 [   right_wall, top_wall], [right_wall, bottom_wall]]).T

    rt60 = 0.3  # seconds
    room_dim = [np.random.uniform(low=6, high=10), np.random.uniform(low=6, high=10)]  # meters

    # We invert Sabine's formula to obtain the parameters for the ISM simulator
    e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)

    # Create the room
    room1 = pra.ShoeBox(
        room_dim, fs=sample_rate, materials=pra.Material(e_absorption), max_order=max_order
    )

    
    voice_loc = [0.0, 0.0]

    # room1 = pra.Room.from_corners(corners,
    #                              fs=sample_rate,
    #                              max_order=10,
    #                              absorption=absorption)

    mic_array1 = generate_mic_array(room1, mic_radius=.07, n_mics=2)
    room1.add_source(voice_loc, signal=audio1)
    room1.image_source_model()
    room1.simulate()

    out1 = room1.mic_array.signals[0, :total_samples]

    # room2 = pra.Room.from_corners(corners,
    #                              fs=sample_rate,
    #                              max_order=10,
    #                              absorption=absorption)
    # Create the room
    room2 = pra.ShoeBox(
        room_dim, fs=sample_rate, materials=pra.Material(e_absorption), max_order=max_order
    )
    mic_array2 = generate_mic_array(room2, mic_radius=.07, n_mics=2)
    room2.add_source(voice_loc, signal=audio2)
    room2.image_source_model()
    room2.simulate()

    out2 = room2.mic_array.signals[1, :total_samples]

    return (out1, out2)


out1, out2 = rir_perturbation(np.ones(500), np.ones(500), 15625)

