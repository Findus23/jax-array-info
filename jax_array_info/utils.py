class Config:
    assign_labels_to_arrays = True


config = Config()  # global singleton object holding the config


def pretty_byte_size(nbytes: int):
    for unit in ("", "Ki", "Mi", "Gi", "Ti"):
        if abs(nbytes) < 1024.0:
            return f"{nbytes:3.1f} {unit}B"
        nbytes /= 1024.0
