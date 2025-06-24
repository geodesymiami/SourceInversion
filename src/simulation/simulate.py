import sys
import VSM_forward


def print_msg(model, x, y, parameters):
    print("#" * 50)
    print(f"Using {model} parameters: {parameters}\n")
    print(f"Grid shape: {x.shape}, {y.shape}\n")

def main(x, y, paramters, **kwargs):
    # Merge paramters and kwargs into a single dictionary
    all_params = {**paramters, **kwargs}

    if 'okada' in all_params["model"]:
        # Extract only the parameters required by the okada function
        required_params = ['xtlc', 'ytlc', 'dtlc', 'length', 'width', 
                           'strike', 'dip', 'param1', 'param2', 
                           'opening', 'opt', 'nu', 'poisson']
        okada_params = {key: float(all_params[key]) for key in required_params if key in all_params}

        print_msg('Okada', x, y, okada_params)

        ux, uy, uz = VSM_forward.okada(x, y, opening=0, opt='R', **okada_params)

    elif 'mogi' in all_params["model"]:
        # Extract only the parameters required by the mogi function
        required_params = ['xcen', 'ycen', 'depth', 'dVol', 'nu']
        mogi_params = {key: float(all_params[key]) for key in required_params if key in all_params}

        print_msg('Mogi', x, y, mogi_params)

        ux, uy, uz = VSM_forward.mogi(x, y, **mogi_params)

    return ux, uy, uz

if __name__ == '__main__':
    args = sys.argv[1:]
    main(*args)