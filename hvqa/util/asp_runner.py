import time
import clingo
from pathlib import Path


class ASPRunner:
    @staticmethod
    def run(temp_file,
            asp_str,
            additional_files=None,
            timeout=5,
            prog_name="",
            opt_mode="optN",
            opt_proven=True):

        temp_file = Path(temp_file)
        f = open(temp_file, "w")
        f.write(asp_str)
        f.close()

        additional_files = [] if additional_files is None else additional_files

        # Add files
        ctl = clingo.Control(message_limit=0)
        ctl.load(str(temp_file))
        [ctl.load(str(f)) for f in additional_files]

        # Configure the solver
        config = ctl.configuration
        config.solve.models = 0
        config.solve.opt_mode = opt_mode

        ctl.ground([("base", [])])

        # Solve ASP
        models = []
        start_time = time.time()
        with ctl.solve(yield_=True) as handle:
            for model in handle:
                if (opt_proven and model.optimality_proven) or not opt_proven:
                    models.append(model.symbols(shown=True))

                if time.time() - start_time > timeout:
                    print(f"WARNING: ASP {prog_name} program reached timeout")
                    handle.cancel()
                    break

        # Cleanup temp file
        temp_file.unlink()

        return models
