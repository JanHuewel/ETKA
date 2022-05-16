from Data import Data
import torch
from globalParams import options
from defaultOptions import fill_tree

def generate_segmented_data(length, segments, seed = [], noise = 0.01, x_start=-10, x_end=10, segment_options = []):

    # define functions for different types of segments
    def seg_const(x, c):
        return c + noise * torch.randn(len(x))
    def seg_lin(x, a, b):
        return a * x + b + noise * torch.randn(len(x))
    def seg_per(x, p, a, p_offset, y_offset):
        return a * torch.sin(x * p + p_offset) + y_offset + noise * torch.randn(len(x))

    segment_length = length/segments
    functions = [seg_const, seg_lin, seg_per]
    x = torch.linspace(x_start,x_end, length)
    y = torch.Tensor()
    for s in range(segments):
        from_index = int(s*segment_length)
        to_index = int((s+1)*segment_length)

        # define type of segment
        try:
            current_seg = ["const", "lin", "per"].index(seed[s])
        except:
            current_seg = int(torch.rand(1).item()*3)

        # grab options for segment
        try:
            current_seg_options = segment_options[s]
        except:
            current_seg_options = dict()
        fill_tree(current_seg_options, options["data generation"][["const", "lin", "per"][current_seg]])

        # fill options so that segments are connected nicely
        if current_seg == 0 and "c" not in current_seg_options:
            if s == 0:
                current_seg_options["c"] = 0
            else:
                current_seg_options["c"] = y[-1].item()
        elif current_seg == 1 and "b" not in current_seg_options:
            if s == 0:
                current_seg_options["b"] = 0
            else:
                current_seg_options["b"] = y[-1].item()
        elif current_seg == 2 and "y_offset" not in current_seg_options:
            if s == 0:
                current_seg_options["y_offset"] = 0
            else:
                current_seg_options["y_offset"] = y[-1].item() - \
                                                  current_seg_options["a"]*\
                                                  torch.sin(torch.tensor(current_seg_options["p_offset"])).item()

        y = torch.cat(y, functions[current_seg](torch.linspace(0,1,to_index-from_index), **segment_options[s]))
    return Data(X=x, Y=y)



def generate_example_data(length, segments=1, seed = [], noise = 0.01):
    segment_length = length/segments
    def type1(x):
        return torch.sin(x * 0.5) + noise*torch.randn(len(x))
    def type2(x):
        return 2 * x + noise*torch.randn(len(x))
    def type3(x):
        return 0.0 * x + noise*torch.randn(len(x))
    functions = [type1, type2, type3]
    x = torch.linspace(-10, 10, length)
    y = torch.Tensor()
    for s in range(segments):
        from_index = int(s*segment_length)
        to_index = int((s+1)*segment_length)
        try:
            y = torch.cat((y,functions[seed[s]](x[from_index:to_index])))
        except:
            r = torch.rand(1).item()
            print(int(r*3))
            y = torch.cat((y,functions[int(r*3)](x[from_index:to_index])))
    return Data(X=x,Y=y)