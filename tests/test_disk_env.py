# %%
import ryenv

MYENV = ryenv.disk_env()
# %%
MYENV.transition((-0.3, -0.2), fps=30)
# %%