"""profsolve.py RUNROOT KEY CELL: cProfile of the product's video solve (core.solve, the solver's own inits) on one
cell's video evidence (multistart.py's setup)."""
import cProfile
import pstats
import sys

exec(open(__file__.replace("profsolve.py", "multistart.py")).read().split("runs = []")[0])
pr = cProfile.Profile()
pr.enable()
r, c = solve_from(None)
pr.disable()
print("end J", round(r.state.J, 1), "CPU", round(c, 1))
st = pstats.Stats(pr)
st.sort_stats("cumulative").print_stats(35)
st.sort_stats("tottime").print_stats(20)
