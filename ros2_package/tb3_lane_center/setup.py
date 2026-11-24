from setuptools import setup

setup(name='tb3_lane_center', version='0.4.0', packages=['tb3_lane_center'],
      data_files=[
          ('share/ament_index/resource_index/packages', ['resource/tb3_lane_center']),
          ('share/tb3_lane_center', ['package.xml']),
          ('share/tb3_lane_center/models', ['models/yellow_line.onnx', 'models/white_line.onnx'])],
      install_requires=['setuptools'],
      entry_points={'console_scripts': ['follow = tb3_lane_center.node:main']})
