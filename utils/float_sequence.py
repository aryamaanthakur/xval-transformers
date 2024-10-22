# The code for encode and decode are from
# https://github.com/facebookresearch/symbolicregression/blob/main/symbolicregression/envs/encoders.py

import numpy as np

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

def get_float_encoder(precision, mantissa_len, max_exponent):
    
    base = (precision + 1) // mantissa_len

    def encode(values):
        """
        Write a float number
        """

        if len(values.shape) == 1:
            seq = []
            value = values
            for val in value:
                if val in [-np.inf, np.inf]:
                    seq.extend(['<pad>']*3)
                    continue
                
                sign = "+" if val >= 0 else "-"
                m, e = (f"%.{precision}e" % val).split("e")
                i, f = m.lstrip("-").split(".")
                i = i + f
                tokens = chunks(i, base)
                expon = int(e) - precision
                if expon < -max_exponent:
                    tokens = ["0" * base] * mantissa_len
                    expon = int(0)
                seq.extend([sign, *["N" + token for token in tokens], "E" + str(expon)])
            return seq
        else:
            seqs = [encode(values[0])]
            N = values.shape[0]
            for n in range(1, N):
                seqs += [encode(values[n])]
        return seqs
    
    return encode

def get_float_decoder(mantissa_len):

    def decode(lst):
        """
        Parse a list that starts with a float.
        Return the float value, and the position it ends in the list.
        """
        if len(lst) == 0:
            return None
        seq = []
        for val in chunks(lst, 2 + mantissa_len):
            for x in val:
                if x[0] not in ["-", "+", "E", "N"]:
                    return np.nan
            try:
                sign = 1 if val[0] == "+" else -1
                mant = ""
                for x in val[1:-1]:
                    mant += x[1:]
                mant = int(mant)
                exp = int(val[-1][1:])
                value = sign * mant * (10 ** exp)
                value = float(value)
            except Exception:
                value = np.nan
            seq.append(value)
        return seq
    
    return decode