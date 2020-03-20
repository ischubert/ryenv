# %%
import ryenv

MYENV = ryenv.disk_env()
# %%
CHANGE = MYENV.transition((-0.3, -0.2), fps=30)
print(CHANGE)
# %%
