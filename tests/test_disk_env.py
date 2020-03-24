# %%
import ryenv

MYENV = ryenv.DiskEnv()
# %%
CHANGE = MYENV.transition((-0.3, -0.2), fps=30)
print(CHANGE)
# %%
