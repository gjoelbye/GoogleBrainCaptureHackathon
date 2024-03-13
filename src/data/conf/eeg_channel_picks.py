
# TODO: Add test to ensure that renamed configurations are compatible with the configurations in conf.eeg_channel_order

hackathon = {
    'original': [
            'Fp1', 'Fp2',
    'F7', 'F3', 'Fz', 'F4', 'F8',
    'T7', 'C3', 'Cz', 'C4', 'T8',
    'P7', 'P3', 'Pz', 'P4', 'P8',
             'O1', 'O2'
    ],
    'renamed': [
            'Fp1', 'Fp2',
    'F7', 'F3', 'Fz', 'F4', 'F8',
    'T7', 'C3', 'Cz', 'C4', 'T8',
    'P7', 'P3', 'Pz', 'P4', 'P8',
             'O1', 'O2'
    ]
}

mmidb = {
    'original': [
            'Fp1.', 'Fp2.',
    'F7..', 'F3..', 'Fz..', 'F4..', 'F8..',
    'T7..', 'C3..', 'Cz..', 'C4..', 'T8..',
    'P7..', 'P3..', 'Pz..', 'P4..', 'P8..',
             'O1..', 'O2..'
    ],
    'renamed': [
            'Fp1', 'Fp2',
    'F7', 'F3', 'Fz', 'F4', 'F8',
    'T7', 'C3', 'Cz', 'C4', 'T8',
    'P7', 'P3', 'Pz', 'P4', 'P8',
             'O1', 'O2'
    ]
}

braincapture = {
    'original': [
                   'FP1', 'FP2',
    'F9', 'F7', 'F3', 'Fz', 'F4', 'F8', 'F10',
    'T9', 'T7', 'C3', 'Cz', 'C4', 'T8', 'T10',
        'TP7',                   'TP8',
    'P9', 'P7', 'P3', 'Pz', 'P4', 'P8', 'P10',
                    'O1', 'O2'
    ],
    'renamed': [
                   'Fp1', 'Fp2',
    'F9', 'F7', 'F3', 'Fz', 'F4', 'F8', 'F10',
    'T9', 'T7', 'C3', 'Cz', 'C4', 'T8', 'T10',
        'TP7',                   'TP8',
    'P9', 'P7', 'P3', 'Pz', 'P4', 'P8', 'P10',
                    'O1', 'O2'
    ]
}


# T7 = T3, T8 = T4, P7 = T5, P8 = T6

tuh_eeg_artefact = {
    'original': [
        'EEG FP1-REF', 'EEG FP2-REF', 
        'EEG F3-REF', 'EEG F4-REF', 
        'EEG C3-REF', 'EEG C4-REF', 
        'EEG P3-REF', 'EEG P4-REF', 
        'EEG O1-REF', 'EEG O2-REF', 
        'EEG F7-REF', 'EEG F8-REF', 
        'EEG T3-REF', 'EEG T4-REF',
        'EEG T5-REF', 'EEG T6-REF', 
        'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF'
    ],
    'renamed': [
        'Fp1', 'Fp2', 
        'F3', 'F4', 
        'C3', 'C4', 
        'P3', 'P4', 
        'O1', 'O2', 
        'F7', 'F8', 
        'T7', 'T8', 
        'P7', 'P8', 
        'Fz', 'Cz', 'Pz'
    ]
}