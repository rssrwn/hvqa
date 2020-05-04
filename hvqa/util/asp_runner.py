import time
import clingo


class ASPRunner:
    def __init__(self, temp_file):
        self.temp_file = temp_file

    def run(self, asp_str, additional_files=None, timeout=5, prog_name=""):
        f = open(self.temp_file, "w")
        f.write(asp_str)
        f.close()

        # Add files
        ctl = clingo.Control(message_limit=0)
        ctl.load(str(self.temp_file))

        # TODO Additional files

        # TODO config (inc optimality, shown, etc)

        # Configure the solver
        config = ctl.configuration
        config.solve.models = 0
        config.solve.opt_mode = "optN"

        ctl.ground([("base", [])])

        # Solve ASP
        models = []
        start_time = time.time()
        with ctl.solve(yield_=True) as handle:
            for model in handle:
                if model.optimality_proven:
                    models.append(model.symbols(shown=True))

                if time.time() - start_time > timeout:
                    print(f"WARNING: ASP {prog_name} program reached timeout")
                    handle.cancel()
                    break

        # Cleanup temp file
        self.temp_file.unlink()

        assert len(models) != 0, f"ASP {prog_name} program is unsatisfiable"

        return models
