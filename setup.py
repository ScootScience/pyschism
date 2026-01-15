#!/usr/bin/env python
import setuptools  # type: ignore[import]


class BuildSchism(setuptools.Command):

  description = "build external SCHISM dependencies"

  user_options = [
    ('url=', None, 'Path for git clone of SCHISM source.'),
    ('branch=', None, 'Branch to use for install'),
    ('hydro=', None, 'Branch to use for install'),
  ]

  def initialize_options(self):
    self.url = None
    self.branch = None
    self.hydro = None

  def finalize_options(self):
    self.url = 'https://github.com/schism-dev/schism' if self.url is None \
      else self.url
    self.branch = 'master' if self.branch is None else self.branch
    self.hydro = True if self.hydro is None else bool(self.hydro)

  def run(self):
    print(self.url, self.branch, self.hydro)
    # subprocess.check_call(["git", "clone", "submodules/jigsaw-python"])
    # # install jigsawpy
    # os.chdir(PARENT / 'submodules/jigsaw-python')
    # subprocess.check_call(["git", "checkout", "master"])
    # self.announce('INSTALLING JIGSAWPY', level=3)
    # subprocess.check_call(["python", "setup.py", "install"])
    # # install jigsaw
    # self.announce(
    #     'INSTALLING JIGSAW LIBRARY AND BINARIES FROM '
    #     'https://github.com/dengwirda/jigsaw-python', level=3)
    # os.chdir("external/jigsaw")
    # os.makedirs("build", exist_ok=True)
    # os.chdir("build")
    # gcc, cpp = self._check_gcc_version()
    # subprocess.check_call(
    #     ["cmake", "..",
    #      "-DCMAKE_BUILD_TYPE=Release",
    #      f"-DCMAKE_INSTALL_PREFIX={PYENV_PREFIX}",
    #      f"-DCMAKE_C_COMPILER={gcc}",
    #      f"-DCMAKE_CXX_COMPILER={cpp}",
    #      ])
    # subprocess.check_call(["make", f"-j{cpu_count()}", "install"])
    # libsaw_prefix = list(PYENV_PREFIX.glob("**/*jigsawpy*")).pop() / '_lib'
    # os.makedirs(libsaw_prefix, exist_ok=True)
    # envlib = PYENV_PREFIX / 'lib' / SYSLIB[platform.system()]
    # os.symlink(envlib, libsaw_prefix / envlib.name)
    # os.chdir(PARENT)
    # subprocess.check_call(
    #   ["git", "submodule", "deinit", "-f", "submodules/jigsaw-python"])


if __name__ == "__main__":
  setuptools.setup(cmdclass={"build_schism": BuildSchism})
