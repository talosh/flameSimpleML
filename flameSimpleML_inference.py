import os
import sys
import time
import queue
import threading
import importlib

import traceback
import atexit
import hashlib
import pickle

import flameSimpleML_framework
importlib.reload(flameSimpleML_framework)
from flameSimpleML_framework import flameAppFramework

try:
    from PySide6 import QtWidgets, QtCore, QtGui
    using_pyside6 = True
except ImportError:
    from PySide2 import QtWidgets, QtCore, QtGui
    using_pyside6 = False

from adsk.libwiretapPythonClientAPI import (
    WireTapClient,
    WireTapServerId,
    WireTapServerHandle,
    WireTapNodeHandle,
    WireTapClipFormat,
    WireTapInt,
    WireTapStr,
)

from pprint import pprint, pformat

class flameSimpleMLInference(QtWidgets.QWidget):

    allEventsProcessed = QtCore.Signal()
    updateInterfaceImage = QtCore.Signal(dict)
    setText = QtCore.Signal(dict)
    setMenu = QtCore.Signal(dict)
    showMessageBox = QtCore.Signal(dict)
    updateFramePositioner = QtCore.Signal()

    class Ui_Progress(object):

        class FlameSlider(QtWidgets.QLineEdit):
            def __init__(self, start_value: int, min_value: int, max_value: int, value_changed_callback = None):
                self.callback = value_changed_callback
                value_is_float = True
                slider_width = 90
                super().__init__()

                # Build slider
                self.setAlignment(QtCore.Qt.AlignCenter)
                self.setMinimumHeight(28)
                self.setMinimumWidth(slider_width)
                self.setMaximumWidth(slider_width)

                if value_is_float:
                    self.spinbox_type = 'Float'
                else:
                    self.spinbox_type = 'Integer'

                self.min = min_value
                self.max = max_value
                self.steps = 1
                self.value_at_press = None
                self.pos_at_press = None
                self.setValue(start_value)
                self.setReadOnly(True)
                self.textChanged.connect(self.value_changed)
                self.setFocusPolicy(QtCore.Qt.NoFocus)
                self.setStyleSheet('QLineEdit {color: rgb(154, 154, 154); background-color: rgb(55, 65, 75); selection-color: rgb(38, 38, 38); selection-background-color: rgb(184, 177, 167); border: none; padding-left: 5px; font: 14px "Discreet"}'
                                'QLineEdit:hover {border: 1px solid rgb(90, 90, 90)}'
                                'QLineEdit:disabled {color: rgb(106, 106, 106); background-color: rgb(55, 65, 75)}'
                                'QToolTip {color: rgb(170, 170, 170); background-color: rgb(71, 71, 71); border: 10px solid rgb(71, 71, 71)}')
                self.clearFocus()

                class Slider(QtWidgets.QSlider):

                    def __init__(self, start_value, min_value, max_value, slider_width):
                        super(Slider, self).__init__()

                        self.setMaximumHeight(4)
                        self.setMinimumWidth(slider_width)
                        self.setMaximumWidth(slider_width)
                        self.setMinimum(min_value)
                        self.setMaximum(max_value)
                        self.setValue(start_value)
                        self.setOrientation(QtCore.Qt.Horizontal)
                        self.setStyleSheet('QSlider {color: rgb(55, 65, 75); background-color: rgb(39, 45, 53)}'
                                        'QSlider::groove {color: rgb(39, 45, 53); background-color: rgb(39, 45, 53)}'
                                        'QSlider::handle:horizontal {background-color: rgb(102, 102, 102); width: 3px}'
                                        'QSlider::disabled {color: rgb(106, 106, 106); background-color: rgb(55, 65, 75)}')
                        self.setDisabled(True)
                        self.raise_()

                def set_slider():
                    slider666.setValue(float(self.text()))

                slider666 = Slider(start_value, min_value, max_value, slider_width)
                self.textChanged.connect(set_slider)

                self.vbox = QtWidgets.QVBoxLayout(self)
                self.vbox.addWidget(slider666)
                self.vbox.setContentsMargins(0, 24, 0, 0)

            def calculator(self):
                from functools import partial

                def clear():
                    calc_lineedit.setText('')

                def button_press(key):

                    if self.clean_line == True:
                        calc_lineedit.setText('')

                    calc_lineedit.insert(key)

                    self.clean_line = False

                def plus_minus():

                    if calc_lineedit.text():
                        calc_lineedit.setText(str(float(calc_lineedit.text()) * -1))

                def add_sub(key):

                    if calc_lineedit.text() == '':
                        calc_lineedit.setText('0')

                    if '**' not in calc_lineedit.text():
                        try:
                            calc_num = eval(calc_lineedit.text().lstrip('0'))

                            calc_lineedit.setText(str(calc_num))

                            calc_num = float(calc_lineedit.text())

                            if calc_num == 0:
                                calc_num = 1
                            if key == 'add':
                                self.setValue(float(self.text()) + float(calc_num))
                            else:
                                self.setValue(float(self.text()) - float(calc_num))

                            self.clean_line = True
                        except:
                            pass

                def enter():

                    if self.clean_line == True:
                        return calc_window.close()

                    if calc_lineedit.text():
                        try:

                            # If only single number set slider value to that number

                            self.setValue(float(calc_lineedit.text()))
                        except:

                            # Do math

                            new_value = calculate_entry()
                            self.setValue(float(new_value))

                    close_calc()
                    
                def equals():

                    if calc_lineedit.text() == '':
                        calc_lineedit.setText('0')

                    if calc_lineedit.text() != '0':

                        calc_line = calc_lineedit.text().lstrip('0')
                    else:
                        calc_line = calc_lineedit.text()

                    if '**' not in calc_lineedit.text():
                        try:
                            calc = eval(calc_line)
                        except:
                            calc = 0

                        calc_lineedit.setText(str(calc))
                    else:
                        calc_lineedit.setText('1')

                def calculate_entry():

                    calc_line = calc_lineedit.text().lstrip('0')

                    if '**' not in calc_lineedit.text():
                        try:
                            if calc_line.startswith('+'):
                                calc = float(self.text()) + eval(calc_line[-1:])
                            elif calc_line.startswith('-'):
                                calc = float(self.text()) - eval(calc_line[-1:])
                            elif calc_line.startswith('*'):
                                calc = float(self.text()) * eval(calc_line[-1:])
                            elif calc_line.startswith('/'):
                                calc = float(self.text()) / eval(calc_line[-1:])
                            else:
                                calc = eval(calc_line)
                        except:
                            calc = 0
                    else:
                        calc = 1

                    calc_lineedit.setText(str(float(calc)))

                    return calc

                def close_calc():
                    calc_window.close()
                    self.setStyleSheet('QLineEdit {color: rgb(154, 154, 154); background-color: rgb(55, 65, 75); selection-color: rgb(154, 154, 154); selection-background-color: rgb(55, 65, 75); border: none; padding-left: 5px; font: 14px "Discreet"}'
                                    'QLineEdit:hover {border: 1px solid rgb(90, 90, 90)}')
                    if self.callback and callable(self.callback):
                        self.callback()

                def revert_color():
                    self.setStyleSheet('QLineEdit {color: rgb(154, 154, 154); background-color: rgb(55, 65, 75); selection-color: rgb(154, 154, 154); selection-background-color: rgb(55, 65, 75); border: none; padding-left: 5px; font: 14px "Discreet"}'
                                    'QLineEdit:hover {border: 1px solid rgb(90, 90, 90)}')
                calc_version = '1.2'
                self.clean_line = False

                calc_window = QtWidgets.QWidget()
                calc_window.setMinimumSize(QtCore.QSize(210, 280))
                calc_window.setMaximumSize(QtCore.QSize(210, 280))
                calc_window.setWindowTitle(f'pyFlame Calc {calc_version}')
                calc_window.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint | QtCore.Qt.Popup)
                calc_window.setAttribute(QtCore.Qt.WA_DeleteOnClose)
                calc_window.destroyed.connect(revert_color)
                calc_window.move(QtGui.QCursor.pos().x() - 110, QtGui.QCursor.pos().y() - 290)
                calc_window.setStyleSheet('background-color: rgb(36, 36, 36)')

                # Labels

                calc_label = QtWidgets.QLabel('Calculator', calc_window)
                calc_label.setAlignment(QtCore.Qt.AlignCenter)
                calc_label.setMinimumHeight(28)
                calc_label.setStyleSheet('color: rgb(154, 154, 154); background-color: rgb(57, 57, 57); font: 14px "Discreet"')

                #  LineEdit

                calc_lineedit = QtWidgets.QLineEdit('', calc_window)
                calc_lineedit.setMinimumHeight(28)
                calc_lineedit.setFocus()
                calc_lineedit.returnPressed.connect(enter)
                calc_lineedit.setStyleSheet('QLineEdit {color: rgb(154, 154, 154); background-color: rgb(55, 65, 75); selection-color: rgb(38, 38, 38); selection-background-color: rgb(184, 177, 167); border: none; padding-left: 5px; font: 14px "Discreet"}')

                # Limit characters that can be entered into lineedit

                regex = QtCore.QRegExp('[0-9_,=,/,*,+,\-,.]+')
                validator = QtGui.QRegExpValidator(regex)
                calc_lineedit.setValidator(validator)

                # Buttons

                def calc_null():
                    # For blank button - this does nothing
                    pass

                class FlameButton(QtWidgets.QPushButton):

                    def __init__(self, button_name, size_x, size_y, connect, parent, *args, **kwargs):
                        super(FlameButton, self).__init__(*args, **kwargs)

                        self.setText(button_name)
                        self.setParent(parent)
                        self.setMinimumSize(size_x, size_y)
                        self.setMaximumSize(size_x, size_y)
                        self.setFocusPolicy(QtCore.Qt.NoFocus)
                        self.clicked.connect(connect)
                        self.setStyleSheet('QPushButton {color: rgb(154, 154, 154); background-color: rgb(58, 58, 58); border: none; font: 14px "Discreet"}'
                                        'QPushButton:hover {border: 1px solid rgb(90, 90, 90)}'
                                        'QPushButton:pressed {color: rgb(159, 159, 159); background-color: rgb(66, 66, 66); border: none}'
                                        'QPushButton:disabled {color: rgb(116, 116, 116); background-color: rgb(58, 58, 58); border: none}')

                blank_btn = FlameButton('', 40, 28, calc_null, calc_window)
                blank_btn.setDisabled(True)
                plus_minus_btn = FlameButton('+/-', 40, 28, plus_minus, calc_window)
                plus_minus_btn.setStyleSheet('color: rgb(154, 154, 154); background-color: rgb(45, 55, 68); font: 14px "Discreet"')
                add_btn = FlameButton('Add', 40, 28, (partial(add_sub, 'add')), calc_window)
                sub_btn = FlameButton('Sub', 40, 28, (partial(add_sub, 'sub')), calc_window)

                #  --------------------------------------- #

                clear_btn = FlameButton('C', 40, 28, clear, calc_window)
                equal_btn = FlameButton('=', 40, 28, equals, calc_window)
                div_btn = FlameButton('/', 40, 28, (partial(button_press, '/')), calc_window)
                mult_btn = FlameButton('/', 40, 28, (partial(button_press, '*')), calc_window)

                #  --------------------------------------- #

                _7_btn = FlameButton('7', 40, 28, (partial(button_press, '7')), calc_window)
                _8_btn = FlameButton('8', 40, 28, (partial(button_press, '8')), calc_window)
                _9_btn = FlameButton('9', 40, 28, (partial(button_press, '9')), calc_window)
                minus_btn = FlameButton('-', 40, 28, (partial(button_press, '-')), calc_window)

                #  --------------------------------------- #

                _4_btn = FlameButton('4', 40, 28, (partial(button_press, '4')), calc_window)
                _5_btn = FlameButton('5', 40, 28, (partial(button_press, '5')), calc_window)
                _6_btn = FlameButton('6', 40, 28, (partial(button_press, '6')), calc_window)
                plus_btn = FlameButton('+', 40, 28, (partial(button_press, '+')), calc_window)

                #  --------------------------------------- #

                _1_btn = FlameButton('1', 40, 28, (partial(button_press, '1')), calc_window)
                _2_btn = FlameButton('2', 40, 28, (partial(button_press, '2')), calc_window)
                _3_btn = FlameButton('3', 40, 28, (partial(button_press, '3')), calc_window)
                enter_btn = FlameButton('Enter', 40, 61, enter, calc_window)

                #  --------------------------------------- #

                _0_btn = FlameButton('0', 89, 28, (partial(button_press, '0')), calc_window)
                point_btn = FlameButton('.', 40, 28, (partial(button_press, '.')), calc_window)

                gridbox = QtWidgets.QGridLayout()
                gridbox.setVerticalSpacing(5)
                gridbox.setHorizontalSpacing(5)

                gridbox.addWidget(calc_label, 0, 0, 1, 4)

                gridbox.addWidget(calc_lineedit, 1, 0, 1, 4)

                gridbox.addWidget(blank_btn, 2, 0)
                gridbox.addWidget(plus_minus_btn, 2, 1)
                gridbox.addWidget(add_btn, 2, 2)
                gridbox.addWidget(sub_btn, 2, 3)

                gridbox.addWidget(clear_btn, 3, 0)
                gridbox.addWidget(equal_btn, 3, 1)
                gridbox.addWidget(div_btn, 3, 2)
                gridbox.addWidget(mult_btn, 3, 3)

                gridbox.addWidget(_7_btn, 4, 0)
                gridbox.addWidget(_8_btn, 4, 1)
                gridbox.addWidget(_9_btn, 4, 2)
                gridbox.addWidget(minus_btn, 4, 3)

                gridbox.addWidget(_4_btn, 5, 0)
                gridbox.addWidget(_5_btn, 5, 1)
                gridbox.addWidget(_6_btn, 5, 2)
                gridbox.addWidget(plus_btn, 5, 3)

                gridbox.addWidget(_1_btn, 6, 0)
                gridbox.addWidget(_2_btn, 6, 1)
                gridbox.addWidget(_3_btn, 6, 2)
                gridbox.addWidget(enter_btn, 6, 3, 2, 1)

                gridbox.addWidget(_0_btn, 7, 0, 1, 2)
                gridbox.addWidget(point_btn, 7, 2)

                calc_window.setLayout(gridbox)

                calc_window.show()

            def value_changed(self):

                # If value is greater or less than min/max values set values to min/max

                if int(self.value()) < self.min:
                    self.setText(str(self.min))
                if int(self.value()) > self.max:
                    self.setText(str(self.max))

            def mousePressEvent(self, event):

                if event.buttons() == QtCore.Qt.LeftButton:
                    self.value_at_press = self.value()
                    self.pos_at_press = event.pos()
                    self.setCursor(QtGui.QCursor(QtCore.Qt.SizeHorCursor))
                    self.setStyleSheet('QLineEdit {color: rgb(217, 217, 217); background-color: rgb(73, 86, 99); selection-color: rgb(154, 154, 154); selection-background-color: rgb(73, 86, 99); border: none; padding-left: 5px; font: 14px "Discreet"}'
                                    'QLineEdit:hover {border: 1px solid rgb(90, 90, 90)}')

            def mouseReleaseEvent(self, event):

                if event.button() == QtCore.Qt.LeftButton:

                    # Open calculator if button is released within 10 pixels of button click

                    if event.pos().x() in range((self.pos_at_press.x() - 10), (self.pos_at_press.x() + 10)) and event.pos().y() in range((self.pos_at_press.y() - 10), (self.pos_at_press.y() + 10)):
                        self.calculator()
                    else:
                        self.setStyleSheet('QLineEdit {color: rgb(154, 154, 154); background-color: rgb(55, 65, 75); selection-color: rgb(154, 154, 154); selection-background-color: rgb(55, 65, 75); border: none; padding-left: 5px; font: 14px "Discreet"}'
                                        'QLineEdit:hover {border: 1px solid rgb(90, 90, 90)}')

                    self.value_at_press = None
                    self.pos_at_press = None
                    self.setCursor(QtGui.QCursor(QtCore.Qt.IBeamCursor))

                    if self.callback and callable(self.callback):
                        self.callback()

                    return

                super().mouseReleaseEvent(event)

            def mouseMoveEvent(self, event):

                if event.buttons() != QtCore.Qt.LeftButton:
                    return

                if self.pos_at_press is None:
                    return

                steps_mult = self.getStepsMultiplier(event)
                delta = event.pos().x() - self.pos_at_press.x()

                if self.spinbox_type == 'Integer':
                    delta /= 10  # Make movement less sensiteve.
                else:
                    delta /= 100
                delta *= self.steps * steps_mult

                value = self.value_at_press + delta
                self.setValue(value)

                super().mouseMoveEvent(event)

            def getStepsMultiplier(self, event):

                steps_mult = 1

                if event.modifiers() == QtCore.Qt.CTRL:
                    steps_mult = 10
                elif event.modifiers() == QtCore.Qt.SHIFT:
                    steps_mult = 0.10

                return steps_mult

            def setMinimum(self, value):

                self.min = value

            def setMaximum(self, value):

                self.max = value

            def setSteps(self, steps):

                if self.spinbox_type == 'Integer':
                    self.steps = max(steps, 1)
                else:
                    self.steps = steps

            def value(self):

                if self.spinbox_type == 'Integer':
                    return int(self.text())
                else:
                    return float(self.text())

            def setValue(self, value):

                if self.min is not None:
                    value = max(value, self.min)

                if self.max is not None:
                    value = min(value, self.max)

                if self.spinbox_type == 'Integer':
                    self.setText(str(int(value)))
                else:
                    # Keep float values to two decimal places

                    self.setText('%.2f' % float(value))

        def setupUi(self, Progress):
            Progress.setObjectName("Progress")
            Progress.setStyleSheet("#Progress {background-color: #242424;} #frame {border: 1px solid #474747; border-radius: 4px;}\n")
                            
            self.verticalLayout = QtWidgets.QVBoxLayout(Progress)
            self.verticalLayout.setSpacing(0)
            self.verticalLayout.setContentsMargins(0, 0, 0, 0)
            self.verticalLayout.setObjectName("verticalLayout")

            # Create a new widget for the stripe at the top
            self.stripe_widget = QtWidgets.QWidget(Progress)
            self.stripe_widget.setStyleSheet("background-color: #474747;")
            self.stripe_widget.setFixedHeight(24)  # Adjust this value to change the height of the stripe

            # Create a label inside the stripe widget
            self.stripe_label = QtWidgets.QLabel(Progress.app_name)  # Replace this with the text you want on the stripe
            self.stripe_label.setStyleSheet("color: #cbcbcb;")  # Change this to set the text color

            # Create a layout for the stripe widget and add the label to it
            stripe_layout = QtWidgets.QHBoxLayout()
            stripe_layout.addWidget(self.stripe_label)
            stripe_layout.addStretch(1)
            stripe_layout.setContentsMargins(18, 0, 0, 0)  # This will ensure the label fills the stripe widget

            # Set the layout to stripe_widget
            self.stripe_widget.setLayout(stripe_layout)

            # Add the stripe widget to the top of the main window's layout
            self.verticalLayout.addWidget(self.stripe_widget)
            self.verticalLayout.addSpacing(4)  # Add a 4-pixel space
            
            '''
            self.src_horisontal_layout = QtWidgets.QHBoxLayout(Progress)
            self.src_horisontal_layout.setSpacing(0)
            self.src_horisontal_layout.setContentsMargins(0, 0, 0, 0)
            self.src_horisontal_layout.setObjectName("srcHorisontalLayout")

            self.src_frame_one = QtWidgets.QFrame(Progress)
            self.src_frame_one.setFrameShape(QtWidgets.QFrame.StyledPanel)
            self.src_frame_one.setFrameShadow(QtWidgets.QFrame.Raised)
            self.src_frame_one.setObjectName("frame")
            self.image_one_label = QtWidgets.QLabel(self.src_frame_one)
            self.image_one_label.setAlignment(QtCore.Qt.AlignCenter)
            frame_one_layout = QtWidgets.QVBoxLayout()
            frame_one_layout.setSpacing(0)
            frame_one_layout.setContentsMargins(0, 0, 0, 0)
            frame_one_layout.addWidget(self.image_one_label)
            self.src_frame_one.setLayout(frame_one_layout)

            self.src_frame_two = QtWidgets.QFrame(Progress)
            self.src_frame_two.setFrameShape(QtWidgets.QFrame.StyledPanel)
            self.src_frame_two.setFrameShadow(QtWidgets.QFrame.Raised)
            self.src_frame_two.setObjectName("frame")
            self.image_two_label = QtWidgets.QLabel(self.src_frame_two)
            self.image_two_label.setAlignment(QtCore.Qt.AlignCenter)
            frame_two_layout = QtWidgets.QVBoxLayout()
            frame_two_layout.setSpacing(0)
            frame_two_layout.setContentsMargins(0, 0, 0, 0)
            frame_two_layout.addWidget(self.image_two_label)
            self.src_frame_two.setLayout(frame_two_layout)

            self.src_horisontal_layout.addWidget(self.src_frame_one)
            self.src_horisontal_layout.addWidget(self.src_frame_two)

            self.verticalLayout.addLayout(self.src_horisontal_layout)
            self.verticalLayout.setStretchFactor(self.src_horisontal_layout, 4)
            '''

            '''
            self.verticalLayout.addSpacing(4)  # Add a 4-pixel space

            self.int_horisontal_layout = QtWidgets.QHBoxLayout(Progress)
            self.int_horisontal_layout.setSpacing(0)
            self.int_horisontal_layout.setContentsMargins(0, 0, 0, 0)
            self.int_horisontal_layout.setObjectName("intHorisontalLayout")

            self.int_frame_1 = QtWidgets.QFrame(Progress)
            self.int_frame_1.setFrameShape(QtWidgets.QFrame.StyledPanel)
            self.int_frame_1.setFrameShadow(QtWidgets.QFrame.Raised)
            self.int_frame_1.setObjectName("frame")
            self.flow1_label = QtWidgets.QLabel(self.int_frame_1)
            self.flow1_label.setAlignment(QtCore.Qt.AlignCenter)
            flow1_layout = QtWidgets.QVBoxLayout()
            flow1_layout.setSpacing(0)
            flow1_layout.setContentsMargins(0, 0, 0, 0)
            flow1_layout.addWidget(self.flow1_label)
            self.int_frame_1.setLayout(flow1_layout)

            self.int_frame_2 = QtWidgets.QFrame(Progress)
            self.int_frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
            self.int_frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
            self.int_frame_2.setObjectName("frame")
            self.flow2_label = QtWidgets.QLabel(self.int_frame_2)
            self.flow2_label.setAlignment(QtCore.Qt.AlignCenter)
            flow2_layout = QtWidgets.QVBoxLayout()
            flow2_layout.setSpacing(0)
            flow2_layout.setContentsMargins(0, 0, 0, 0)
            flow2_layout.addWidget(self.flow2_label)
            self.int_frame_2.setLayout(flow2_layout)

            self.int_frame_3 = QtWidgets.QFrame(Progress)
            self.int_frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
            self.int_frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
            self.int_frame_3.setObjectName("frame")
            self.flow3_label = QtWidgets.QLabel(self.int_frame_3)
            self.flow3_label.setAlignment(QtCore.Qt.AlignCenter)
            flow3_layout = QtWidgets.QVBoxLayout()
            flow3_layout.setSpacing(0)
            flow3_layout.setContentsMargins(0, 0, 0, 0)
            flow3_layout.addWidget(self.flow3_label)
            self.int_frame_3.setLayout(flow3_layout)

            self.int_frame_4 = QtWidgets.QFrame(Progress)
            self.int_frame_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
            self.int_frame_4.setFrameShadow(QtWidgets.QFrame.Raised)
            self.int_frame_4.setObjectName("frame")
            self.flow4_label = QtWidgets.QLabel(self.int_frame_4)
            self.flow4_label.setAlignment(QtCore.Qt.AlignCenter)
            flow4_layout = QtWidgets.QVBoxLayout()
            flow4_layout.setSpacing(0)
            flow4_layout.setContentsMargins(0, 0, 0, 0)
            flow4_layout.addWidget(self.flow4_label)
            self.int_frame_4.setLayout(flow4_layout)

            self.int_horisontal_layout.addWidget(self.int_frame_1)
            self.int_horisontal_layout.addWidget(self.int_frame_2)
            self.int_horisontal_layout.addWidget(self.int_frame_3)
            self.int_horisontal_layout.addWidget(self.int_frame_4)

            self.verticalLayout.addLayout(self.int_horisontal_layout)
            self.verticalLayout.setStretchFactor(self.int_horisontal_layout, 2)

            self.verticalLayout.addSpacing(4)  # Add a 4-pixel space

            '''

            self.res_frame = QtWidgets.QFrame(Progress)
            self.res_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
            self.res_frame.setFrameShadow(QtWidgets.QFrame.Raised)
            self.res_frame.setObjectName("frame")
            self.image_res_label = QtWidgets.QLabel(self.res_frame)
            self.image_res_label.setAlignment(QtCore.Qt.AlignCenter)
            frame_res_layout = QtWidgets.QVBoxLayout()
            frame_res_layout.setSpacing(0)
            frame_res_layout.setContentsMargins(8, 8, 8, 8)
            frame_res_layout.addWidget(self.image_res_label)
            self.res_frame.setLayout(frame_res_layout)

            self.verticalLayout.addWidget(self.res_frame)
            self.verticalLayout.setStretchFactor(self.res_frame, 8)

            self.verticalLayout.addSpacing(4)  # Add a 4-pixel space

            # Create a new horizontal layout for the bottom of the window
            bottom_layout = QtWidgets.QHBoxLayout()

            # Add a close button to the bottom layout
            self.close_button = QtWidgets.QPushButton("Close")
            self.close_button.clicked.connect(Progress.close_application)
            self.close_button.setContentsMargins(10, 4, 10, 4)
            self.set_button_style(self.close_button)
            bottom_layout.addWidget(self.close_button, alignment=QtCore.Qt.AlignLeft)

            # Add spacer
            spacer = QtWidgets.QLabel('', Progress)
            spacer.setMinimumWidth(4)
            bottom_layout.addWidget(spacer, alignment=QtCore.Qt.AlignLeft)

            # StartFrame label
            self.cur_frame_label = QtWidgets.QLabel(' ', Progress)
            self.cur_frame_label.setMinimumWidth(48)
            self.cur_frame_label.setContentsMargins(10, 0, 10, 0)
            self.cur_frame_label.setStyleSheet(
                'QLabel {color: rgb(154, 154, 154); background-color: #292929; border: 1px solid #474747; font: 14px "Discreet";}'
                )
            self.cur_frame_label.setAlignment(QtCore.Qt.AlignCenter)
            bottom_layout.addWidget(self.cur_frame_label, alignment=QtCore.Qt.AlignLeft)

            # Info label
            self.info_label = QtWidgets.QLabel('Frame:', Progress)
            self.info_label.setContentsMargins(10, 4, 10, 4)
            self.info_label.setStyleSheet("color: #cbcbcb;")
            bottom_layout.addWidget(self.info_label)
            bottom_layout.setStretchFactor(self.info_label, 1)

            # EndFrame label
            self.end_frame_label = QtWidgets.QLabel(' ', Progress)
            self.end_frame_label.setMinimumWidth(48)
            self.end_frame_label.setContentsMargins(10, 0, 10, 0)
            self.end_frame_label.setStyleSheet(
                'QLabel {color: rgb(154, 154, 154); background-color: #292929; border: 1px solid #474747; font: 14px "Discreet";}'
                )
            self.end_frame_label.setAlignment(QtCore.Qt.AlignCenter)
            bottom_layout.addWidget(self.end_frame_label)
            bottom_layout.addSpacing(4)

            '''
            # Tiles selector button
            self.tiles_selector = QtWidgets.QPushButton('Tiles')
            self.tiles_selector.setContentsMargins(10, 4, 10, 4)
            self.set_cpu_button_style(self.tiles_selector)
            bottom_layout.addWidget(self.tiles_selector, alignment=QtCore.Qt.AlignRight)
            bottom_layout.addSpacing(4)

            # CPU selector button
            self.cpu_selector = QtWidgets.QPushButton('CPU')
            self.cpu_selector.setContentsMargins(10, 4, 10, 4)
            self.set_cpu_button_style(self.cpu_selector)
            bottom_layout.addWidget(self.cpu_selector, alignment=QtCore.Qt.AlignRight)
            bottom_layout.addSpacing(4)
            '''

            '''
            # TW Speed test field:
            if Progress.tw_speed:
                self.tw_speed_input = self.FlameSlider(Progress.tw_speed, -9999, 9999, Progress.on_SpeedValueChange)
                self.tw_speed_input.setContentsMargins(4, 0, 0, 0)
                bottom_layout.addWidget(self.tw_speed_input, alignment=QtCore.Qt.AlignRight)
                bottom_layout.addSpacing(4)
            '''

            '''
            # mode selector button
            current_mode = Progress.parent_app.current_mode
            modes = Progress.parent_app.modes
            mode_selector_text = modes.get(current_mode, sorted(modes.keys())[0])
            self.mode_selector = QtWidgets.QPushButton(mode_selector_text)
            self.mode_selector.setContentsMargins(10, 4, 10, 4)
            self.set_selector_button_style(self.mode_selector)
            self.mode_selector.setMinimumSize(QtCore.QSize(120, 28))
            # self.mode_selector.setMaximumSize(QtCore.QSize(120, 28))
            bottom_layout.addWidget(self.mode_selector, alignment=QtCore.Qt.AlignRight)
            bottom_layout.addSpacing(4)
            '''

            # Model selector button
            self.model_selector = QtWidgets.QPushButton('Load Model ...')
            self.model_selector.setContentsMargins(10, 4, 10, 4)
            self.set_selector_button_style(self.model_selector)
            bottom_layout.addWidget(self.model_selector, alignment=QtCore.Qt.AlignRight)
            bottom_layout.addSpacing(4)

            # Render button
            self.render_button = QtWidgets.QPushButton("Render")
            self.render_button.clicked.connect(Progress.toggle_render_button)
            self.render_button.setContentsMargins(4, 4, 10, 4)
            self.set_button_style(self.render_button)
            bottom_layout.addWidget(self.render_button, alignment=QtCore.Qt.AlignRight)

            # Add the bottom layout to the main layout
            self.verticalLayout.addLayout(bottom_layout)

            self.retranslateUi(Progress)
            QtCore.QMetaObject.connectSlotsByName(Progress)

        def retranslateUi(self, Progress):
            Progress.setWindowTitle("Form")
            # self.progress_header.setText("Timewarp ML")
            # self.progress_message.setText("Reading images....")

        def set_button_style(self, button):
            button.setMinimumSize(QtCore.QSize(150, 28))
            button.setMaximumSize(QtCore.QSize(150, 28))
            button.setFocusPolicy(QtCore.Qt.NoFocus)
            button.setStyleSheet('QPushButton {color: rgb(154, 154, 154); background-color: rgb(58, 58, 58); border: none; font: 14px}'
            'QPushButton:hover {border: 1px solid rgb(90, 90, 90)}'
            'QPushButton:pressed {color: rgb(159, 159, 159); background-color: rgb(66, 66, 66); border: 1px solid rgb(90, 90, 90)}'
            'QPushButton:disabled {color: rgb(116, 116, 116); background-color: rgb(58, 58, 58); border: none}'
            'QPushButton::menu-indicator {subcontrol-origin: padding; subcontrol-position: center right}'
            'QToolTip {color: rgb(170, 170, 170); background-color: rgb(71, 71, 71); border: 10px solid rgb(71, 71, 71)}')

        def set_cpu_button_style(self, button):
            button.setMinimumSize(QtCore.QSize(80, 28))
            button.setMaximumSize(QtCore.QSize(80, 28))
            button.setFocusPolicy(QtCore.Qt.NoFocus)
            button.setStyleSheet('QPushButton {color: rgb(154, 154, 154); background-color: rgb(58, 58, 58); border: none; font: 14px}'
            'QPushButton:hover {border: 1px solid rgb(90, 90, 90)}'
            'QPushButton:pressed {color: rgb(159, 159, 159); background-color: rgb(66, 66, 66); border: 1px solid rgb(90, 90, 90)}'
            'QPushButton:disabled {color: rgb(116, 116, 116); background-color: rgb(58, 58, 58); border: none}'
            'QPushButton::menu-indicator {subcontrol-origin: padding; subcontrol-position: center right}'
            'QToolTip {color: rgb(170, 170, 170); background-color: rgb(71, 71, 71); border: 10px solid rgb(71, 71, 71)}')

        def set_selector_button_style(self, button):
            button.setMinimumSize(QtCore.QSize(190, 28))
            # button.setMaximumSize(QtCore.QSize(150, 28))
            button.setFocusPolicy(QtCore.Qt.NoFocus)
            button.setStyleSheet('QPushButton {color: rgb(154, 154, 154); background-color: rgb(44, 54, 68); border: none; font: 14px}'
            'QPushButton:hover {border: 1px solid rgb(90, 90, 90)}'
            'QPushButton:pressed {color: rgb(159, 159, 159); background-color: rgb(44, 54, 68); border: 1px solid rgb(90, 90, 90)}'
            'QPushButton:disabled {color: rgb(116, 116, 116); background-color: rgb(58, 58, 58); border: none}'
            'QPushButton::menu-indicator {image: none;}')

    def __init__(self, *args, **kwargs):
        super().__init__()

        self.using_pyside6 = using_pyside6

        self.name = self.__class__.__name__
        self.selection = kwargs.get('selection')

        self.settings = kwargs.get('settings', dict())
        self.fw = flameAppFramework(settings = self.settings)
        self.app_name = self.fw.app_name
        self.log = self.fw.log
        self.debug = self.fw.debug
        self.log_debug = self.fw.log_debug
        self.version = self.settings.get('version', 'UnknownVersion')
        self.temp_folder = self.fw.temp_folder
        self.temp_library = None

        self.prefs = self.fw.prefs_dict(self.fw.prefs, self.name)
        self.prefs_user = self.fw.prefs_dict(self.fw.prefs_user, self.name)
        self.prefs_global = self.fw.prefs_dict(self.fw.prefs_global, self.name)

        self.prefs['version'] = self.version
        self.prefs_user['version'] = self.version
        self.prefs_global['version'] = self.version
        self.fw.save_prefs()

        self.model_state_dict_path = None
        current_model_index = self.prefs.get('current_model_index')
        if current_model_index:
            model_menu_items = self.prefs.get('recent_models')
            if model_menu_items:
                self.model_state_dict_path = model_menu_items.get(current_model_index)

        self.model_state_dict = {}
        self.models_folder = os.path.join(
            os.path.dirname(__file__),
            'models'
        )
        self.models = {}
        self.current_model = None
        self.torch = None
        self.torch_device = 'cpu'
        
        self.message_queue = queue.Queue()
        self.ui_images_queue = queue.Queue(maxsize=9)
        self.frames_to_save_queue = queue.Queue(maxsize=9)

        self.min_frame = 1
        self.max_frame = 99
        self.current_frame = 1
        self.input_channels = 3
        self.output_channels = 3

        self.destination_node_id = ''

        # mouse position on a press event
        self.mousePressPos = None

        self.frame_thread = None
        self.rendering = False

        #### APP STATE MEGA DICT ####

        self.current_state = {
            'view_mode': 'F4',
            'rendering_by_render_button': False,
            'src_image_data': None,
            'res_image_data': None
        }

        #### THREADS INITIALIZAION ####

        self.threads = True

        self.log_debug('starting message thread')
        # set up message thread
        self.message_thread = threading.Thread(target=self.process_messages)
        self.message_thread.daemon = True
        self.message_thread.start()
        self.log_debug('message thread started')

        self.log_debug('starting interface image thread')
        # set up message thread
        self.interface_image_thread = threading.Thread(target=self.process_interface_images)
        self.interface_image_thread.daemon = True
        self.interface_image_thread.start()
        self.log_debug('interface image thread started')

        self.log_debug('starting frame save thread')
        # set up save thread
        self.save_thread = threading.Thread(target=self.process_frames_to_save)
        self.save_thread.daemon = True
        self.save_thread.start()
        self.log_debug('frame save thread started')

        self.render_loop_thread = None

        #### UI INITIALIZATION ####

        # A flag to check if all events have been processed
        self.allEventsFlag = False
        # Connect signals to slots
        self.allEventsProcessed.connect(self.on_allEventsProcessed)
        self.updateInterfaceImage.connect(self.on_UpdateInterfaceImage)
        self.setText.connect(self.on_setText)
        self.setMenu.connect(self.on_setMenu)
        self.showMessageBox.connect(self.on_showMessageBox)
        self.updateFramePositioner.connect(self.update_frame_positioner)

        # load in the UI
        self.log_debug('Initializing UI')
        self.ui = self.Ui_Progress()
        self.log_debug('Loading SetupUI')
        self.ui.setupUi(self)
        self.log_debug('Loaded')

        self.ui.info_label.setText('Initializing...')

        # set window flags
        self.setWindowFlags(
            # QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint
            # QtCore.Qt.Window | QtCore.Qt.Tool | QtCore.Qt.FramelessWindowHint
            QtCore.Qt.Window | QtCore.Qt.Tool
        )

        # calculate window dimentions
        try:
            W = self.selection[0].width
            H = self.selection[0].height
        except:
            W = 1280
            H = 720

        self.log_debug(f'using PySide6: {self.using_pyside6}')

        def get_screen_geometry():
            if self.using_pyside6:
                screen = QtGui.QGuiApplication.primaryScreen()
                screen_geometry = screen.geometry()
            else:
                desktop = QtWidgets.QApplication.desktop()
                screen_geometry = desktop.screenGeometry(desktop.primaryScreen())
            return screen_geometry


        # desktop = QtWidgets.QApplication.desktop()
        # screen_geometry = desktop.screenGeometry(desktop.primaryScreen())
        screen_geometry = get_screen_geometry()

        max_width = screen_geometry.width() * 0.88
        max_height = screen_geometry.height() * 0.88

        desired_width = W
        # Coeeficient to accomodate additional rows: 
        # (1 + 1/n) * H + ttile_h + title_spacing + lower_stripe_h + lower_stripe_spacing
        # desired_height = (1 + (1/4)) * H + (24 + 18 + 28 + 10)
        desired_height = H + (24 + 18 + 28 + 10) 
                                                        
        scale_factor = min(max_width / desired_width, max_height / desired_height)
        scaled_width = desired_width * scale_factor
        scaled_height = desired_height * scale_factor

        # Check that scaled_width is not less than the minimum
        if scaled_width < 1024:
            scaled_width = 1024

        # Set window dimensions
        self.setGeometry(0, 0, scaled_width, scaled_height)

        # Move the window to the center of the screen
        screen_center = screen_geometry.center()
        self.move(screen_center.x() - scaled_width // 2, screen_center.y() - scaled_height // 2 - 100)

        # show window and fix its size
        self.setWindowTitle(self.app_name + ' ' + self.version)
        self.show()
        self.raise_()
        self.activateWindow()
        self.setFixedSize(self.size())

        QtCore.QTimer.singleShot(99, self.after_show)

    def after_show(self):
        self.message_queue.put({'type': 'info', 'message': 'Checking requirements...'})
        self.processEvents()
        missing_requirements = self.fw.check_requirements(self.fw.requirements)

        sys.exit()


        if missing_requirements:
            self.message_queue.put({'type': 'info', 'message': 'Requirements check failed'})
            python_executable_path = sys.executable
            try:
                import flame
                flame_version = flame.get_version()
                python_executable_path = f'/opt/Autodesk/python/{flame_version}/bin/python'
            except:
                pass

            missing_req_string = '\n' + ', \n'.join(missing_requirements)
            message_string = f'Unable to import:\n{missing_req_string}\n\n'
            message_string += f"Make sure reqiured packages are available to Flame's built-in python interpreter.\n\n"
            message_string += f'To install manually use:\n"{python_executable_path} -m pip install <package-name>"'
            self.message_queue.put(
                {'type': 'mbox',
                'message': message_string,
                'action': self.close_application}
            )
            return
        
        self.torch_device = self.set_device()

        self.clips_parent = self.selection[0].parent
        duration = self.selection[0].duration.frame
        relative_start_frame = self.selection[0].start_time.get_value().relative_frame

        for selected_item in self.selection:
            if selected_item.duration.frame != duration:
                message_string = 'Please select input clips of the same length'
                message_string += ' or use "mark in" and "mark out" to give segments of the same duration'
                self.message_queue.put(
                    {'type': 'mbox',
                    'message': message_string,
                    'action': self.close_application}
                )
                return

        self.min_frame = relative_start_frame
        self.max_frame = relative_start_frame + duration - 1
        self.message_queue.put(
            {'type': 'setText',
            'widget': 'cur_frame_label',
            'text': str(self.min_frame)}
        )
        self.message_queue.put(
            {'type': 'setText',
            'widget': 'end_frame_label',
            'text': str(self.max_frame)}
        )

        self.message_queue.put({'type': 'info', 'message': 'Scanning for models...'})
        try:
            self.models = self.scan_models(self.models_folder)
        except Exception as e:
            message_string = f'Unable to load models from {self.models_folder}:\n"{e}"'
            self.message_queue.put(
                {'type': 'mbox',
                'message': message_string,
                'action': None}
            )
        self.message_queue.put({'type': 'info', 'message': f'Loaded {len(self.models.keys())} models'})

        self.frames_map = self.compose_frames_map(self.selection)

        self.message_queue.put({'type': 'info', 'message': 'Creating destination shared library...'})
        self.create_temp_library(self.selection)
        self.message_queue.put({'type': 'info', 'message': 'Creating destination clip node...'})
        self.destination_node_id = self.create_destination_node(
            self.selection[0],
            duration
            )

        print (f'total channels: {self.get_total_channels_number(self.selection)}')

        self.fill_model_menu()

        if self.model_state_dict_path:
            self.load_model_state_dict(self.model_state_dict_path)
            if self.model_state_dict:
                self.load_model(self.model_state_dict)

        self.message_queue.put({'type': 'info', 'message': 'Reading source clip(s)...'})
        self.set_current_frame(self.min_frame)

        # self.frames_map = self.compose_frames_map(self.selection)

        '''
        self.parent_app.torch_device = self.set_torch_device()

        self.message_queue.put({'type': 'info', 'message': 'Creating destination shared library...'})
        self.processEvents()
        self.parent_app.create_temp_library(self.selection)
        if not self.parent_app.temp_library:
            return

        self.processEvents()
        self.message_queue.put({'type': 'info', 'message': 'Building frames map...'})
        self.processEvents()
        self.frames_map = self.parent_app.compose_frames_map(self.selection, self.mode)

        # print (f'frames map: {pformat(self.frames_map)}')

        self.min_frame = min(self.frames_map.keys())
        self.max_frame = max(self.frames_map.keys())
        self.message_queue.put(
            {'type': 'setText',
            'widget': 'cur_frame_label',
            'text': str(self.min_frame)}
        )
        self.message_queue.put(
            {'type': 'setText',
            'widget': 'end_frame_label',
            'text': str(self.max_frame)}
        )

        

        self.message_queue.put({'type': 'info', 'message': 'Creating destination clip node...'})
        self.processEvents()
        self.destination_node_id = self.parent_app.create_destination_node(
            self.selection,
            len(self.frames_map.keys())
            )
        if not self.destination_node_id:
            return

        self.message_queue.put({'type': 'info', 'message': 'Reading source clip(s)...'})
        self.processEvents()
        
        self.set_current_frame(self.min_frame)
        '''

        '''
        self.frame_thread = threading.Thread(target=self._process_current_frame, kwargs={'single_frame': True})
        self.frame_thread.daemon = True
        self.frame_thread.start()
        '''

    def processEvents(self):
        try:
            QtWidgets.QApplication.instance().processEvents()
            self.allEventsProcessed.emit()
            while not self.allEventsFlag:
                time.sleep(1e-9)
        except:
            pass

    def mousePressEvent(self, event):
        child = self.childAt(event.pos())
        if child == self.ui.info_label:
            self.mousePressPos = None
            super().mousePressEvent(event)
            return
        elif child == self.ui.cur_frame_label:
            self.set_current_frame(self.min_frame, render = False)
            self.mousePressPos = None
            super().mousePressEvent(event)
            return
        elif child == self.ui.end_frame_label:
            self.set_current_frame(self.max_frame, render = False)
            self.mousePressPos = None
            super().mousePressEvent(event)
            return

        # Record the position at which the mouse was pressed.
        self.mousePressPos = event.globalPos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        child = self.childAt(event.pos())
        if child == self.ui.info_label:
            relative_pos = self.ui.info_label.mapFromParent(event.pos())
            label_width = self.ui.info_label.width()
            new_frame = int(self.min_frame + (relative_pos.x() - 0) * (self.max_frame - self.min_frame) / (label_width - 0))
            self.set_current_frame(new_frame, render = False)
            super().mouseReleaseEvent(event)
            return
        
        if self.mousePressPos is not None:
            # Calculate the new position of the window.
            newPos = self.pos() + (event.globalPos() - self.mousePressPos)
            # Move the window to the new position.
            self.move(newPos)
            # Update the position at which the mouse was pressed.
            self.mousePressPos = event.globalPos()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self.mousePressPos = None
        child = self.childAt(event.pos())
        if child == self.ui.info_label:
            relative_pos = self.ui.info_label.mapFromParent(event.pos())
            label_width = self.ui.info_label.width()
            new_frame = int(self.min_frame + (relative_pos.x() - 0) * (self.max_frame - self.min_frame) / (label_width - 0))
            # print(f"Clicked on label at position: {relative_pos.x()}, {relative_pos.y()}")
            # print (f'frame: {new_frame}')
            self.set_current_frame(new_frame)

        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Left:
            self.left_arrow_pressed()
        elif event.key() == QtCore.Qt.Key_Right:
            self.right_arrow_pressed()
        elif event.key() == QtCore.Qt.Key_F1:
            self.f1_key_pressed()
        elif event.key() == QtCore.Qt.Key_F4:
            self.f4_key_pressed()
        else:
            super().keyPressEvent(event)  # Pass the event to the parent's handler

    def left_arrow_pressed(self):
        if self.current_state.get('rendering_by_render_button'):
            return
        self.set_current_frame(self.current_frame - 1 if self.current_frame > self.min_frame else self.min_frame)

    def right_arrow_pressed(self):
        if self.current_state.get('rendering_by_render_button'):
            return
        self.set_current_frame(self.current_frame + 1 if self.current_frame < self.max_frame else self.max_frame)

    def f1_key_pressed(self):
        self.current_state['view_mode'] = 'F1'
        print (self.current_state['view_mode'])
        if self.current_state.get('rendering_by_render_button'):
            return
        
        if self.current_state.get('src_image_data') is not None:
            self.update_interface_image_torch(
                    self.current_state.get('src_image_data')[:, :, :3],
                    self.ui.image_res_label,
                    text = 'Frame: ' + str(self.current_frame)
                )

    def f4_key_pressed(self):
        self.current_state['view_mode'] = 'F4'
        print (self.current_state['view_mode'])

        if self.current_state.get('rendering_by_render_button'):
            return

        if self.current_state.get('res_image_data') is not None:
            self.update_interface_image_torch(
                    self.current_state.get('res_image_data')[:, :, :3],
                    self.ui.image_res_label,
                    text = 'Frame: ' + str(self.current_frame)
                )

    def on_allEventsProcessed(self):
        self.allEventsFlag = True

    def on_UpdateInterfaceImage(self, item):
        import torch

        if torch.is_tensor(item.get('image')):
            self._update_interface_image_torch(
                item.get('image'),
                item.get('image_label'),
                item.get('text')
            )
        else:
            self._update_interface_image(
                item.get('image'),
                item.get('image_label'),
                item.get('text')
            )

    def on_setText(self, item):
        widget_name = item.get('widget', 'unknown')
        text = item.get('text', 'unknown')
        if hasattr(self.ui, widget_name):
            getattr(self.ui, widget_name).setText(text)
        self.processEvents()

    def on_setMenu(self, item):
        widget_name = item.get('widget', 'unknown')
        menu = item.get('menu', 'unknown')
        if hasattr(self.ui, widget_name):
            getattr(self.ui, widget_name).setMenu(menu)
        self.processEvents()

    def on_showMessageBox(self, item):
        message = item.get('message')
        action = item.get('action', None)

        mbox = QtWidgets.QMessageBox()
        mbox.setWindowFlags(QtCore.Qt.Tool)
        mbox.setWindowTitle(self.app_name)
        mbox.setStyleSheet("""
            QMessageBox {
                background-color: #313131;
                color: #9a9a9a;
                text-align: center;
            }
            QMessageBox QPushButton {
                width: 80px;
                height: 24px;
                color: #9a9a9a;
                background-color: #424142;
                border-top: 1px inset #555555;
                border-bottom: 1px inset black
            }
            QMessageBox QPushButton:pressed {
                font:italic;
                color: #d9d9d9
            }
        """)

        mbox.setText(message)
        mbox.exec_()

        if action and callable(action):
            action()

    def update_interface_image(self, array, image_label, text = None):
        if self.ui_images_queue.qsize() > 8:
            return
        
        item = {
            'type': 'image',
            'image': array,
            'image_label': image_label,
            'text': text
        }
        self.ui_images_queue.put(item)

    def _update_interface_image(self, array, image_label, text = None):
        import numpy as np
        import torch

        if array is None:
            image_label.clear()
            return
        
        if isinstance(array, torch.Tensor):
            # colourmanagement should go here
            if (array.dtype == torch.float16) or (array.dtype == torch.float32):
                img_torch = torch.clip(array, 0, 1) * 255
                img_torch = img_torch.byte()
                img = img_torch.cpu().detach().numpy()
                del img_torch
            else:
                img_torch = array.byte()
                img = img_torch.cpu().detach().numpy()
                del img_torch
        else:
            # colourmanagement should go here
            if (array.dtype == np.float16) or (array.dtype == np.float32):
                img = np.clip(array, 0, 1) * 255
                img = img.astype(np.uint8)
            else:
                img = array.astype(np.uint8)
        
        img_contiguous = img.copy(order='C')

        # Convert the numpy array to a QImage
        height, width, _ = img_contiguous.shape
        bytes_per_line = 3 * width
        qt_image = QtGui.QImage(img_contiguous.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
        qt_pixmap = QtGui.QPixmap.fromImage(qt_image)
        parent_frame = image_label.parent()
        scaled_pixmap = qt_pixmap.scaled(
            parent_frame.size() * 0.9, 
            QtCore.Qt.KeepAspectRatio, 
            QtCore.Qt.SmoothTransformation)
        if text:
            margin = 4
            origin_x = 2
            origin_y = 2

            painter = QtGui.QPainter(scaled_pixmap)
            font = QtGui.QFont("Discreet", 12)
            painter.setFont(font)
            
            '''
            metrics = QtGui.QFontMetrics(font)
            text_width = metrics.horizontalAdvance(text)
            text_height = metrics.height()
            rect_x = origin_x
            rect_y = scaled_pixmap.height() - text_height - margin * 2 - origin_y
            rect_width = text_width + margin * 2 + 2
            rect_height = text_height + margin * 2
            color = QtGui.QColor(0, 0, 0)
            radius = 2
            painter.setBrush(color)
            painter.setOpacity(0.2)
            painter.drawRoundedRect(rect_x, rect_y, rect_width, rect_height, radius, radius)
            '''

            painter.setOpacity(1.0)
            painter.setPen(QtGui.QColor(255, 255, 255))
            text_x = margin + origin_x
            text_y = scaled_pixmap.height() - margin -origin_y
            painter.drawText(text_x, text_y, text)
            painter.end()

        image_label.setPixmap(scaled_pixmap)
        self.processEvents()

        del img
        del img_contiguous
        del qt_pixmap

        '''
        QtWidgets.QApplication.instance().processEvents()
        time.sleep(0.001)
        image_label.setPixmap(scaled_pixmap)
        QtWidgets.QApplication.instance().processEvents()
        '''

    def update_interface_image_torch(self, array, image_label, text = None):
        import torch
        import torch.nn.functional as F

        if self.message_queue.qsize() > 8:
            return

        if array is None:
            image_label.clear()
            return

        label_size = image_label.size()
        h, w, d = array.shape
        scale_factor = min((0.99 * label_size.height())/h, (0.99 * label_size.width())/w)
        array = array.permute(2, 0, 1).unsqueeze(0)
        array = F.interpolate(array, scale_factor=scale_factor, mode="bilinear", align_corners=False)
        array = array.squeeze(0).permute(1, 2, 0)

        item = {
            'type': 'image',
            'image': array,
            'image_label': image_label,
            'text': text
        }

        self.ui_images_queue.put(item)

    def _update_interface_image_torch(self, array, image_label, text = None):
        import torch
        import numpy as np

        if array is None:
            image_label.clear()
            return
        
        if (array.dtype == torch.float16) or (array.dtype == torch.float32):
            img_torch = torch.clip(array, 0, 1) * 255
            img_torch = img_torch.byte()
            img = img_torch.cpu().detach().numpy()
            del img_torch
        else:
            img_torch = array.byte()
            img = img_torch.cpu().detach().numpy()
            del img_torch

        img_contiguous = np.ascontiguousarray(img, dtype=np.uint8)
        del img
        height, width, _ = img_contiguous.shape
        bytes_per_line = 3 * width
        qt_image = QtGui.QImage(img_contiguous.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
        qt_pixmap = QtGui.QPixmap.fromImage(qt_image)
        del img_contiguous
        del qt_image

        if text:
            margin = 4
            origin_x = 2
            origin_y = 2

            painter = QtGui.QPainter(qt_pixmap)
            font = QtGui.QFont("Discreet", 12)
            painter.setFont(font)
            
            '''
            metrics = QtGui.QFontMetrics(font)
            text_width = metrics.horizontalAdvance(text)
            text_height = metrics.height()
            rect_x = origin_x
            rect_y = scaled_pixmap.height() - text_height - margin * 2 - origin_y
            rect_width = text_width + margin * 2 + 2
            rect_height = text_height + margin * 2
            color = QtGui.QColor(0, 0, 0)
            radius = 2
            painter.setBrush(color)
            painter.setOpacity(0.2)
            painter.drawRoundedRect(rect_x, rect_y, rect_width, rect_height, radius, radius)
            '''

            painter.setOpacity(1.0)
            painter.setPen(QtGui.QColor(255, 255, 255))
            text_x = margin + origin_x
            text_y = qt_pixmap.height() - margin -origin_y
            painter.drawText(text_x, text_y, text)
            painter.end()

        image_label.setPixmap(qt_pixmap)
        self.processEvents()
        del qt_pixmap

    def update_frame_positioner(self):
        import numpy as np

        label_width = self.ui.info_label.width()
        label_height = self.ui.info_label.height()
        margin = 4

        # map x1 from [x,y] to [m, n]: m1 = m + (x1 - x) * (n - m) / (y - x)
        marker_pos = 4 + (self.current_frame - self.min_frame) * (label_width - 8) / (self.max_frame - self.min_frame)
        if marker_pos < margin:
            marker_pos = margin
        elif marker_pos > label_width - margin:
            marker_pos = label_width - margin
        bg = np.full((1, label_width, 3), [36, 36, 36], dtype=np.uint8)
        bg[0, int(marker_pos), :] = [135, 122, 28]
        bg = np.repeat(bg, label_height, axis=0)
        qt_image = QtGui.QImage(bg.data, label_width, label_height, 3 * label_width, QtGui.QImage.Format_RGB888)
        qt_pixmap = QtGui.QPixmap.fromImage(qt_image)
        palette = QtGui.QPalette()
        palette.setBrush(QtGui.QPalette.Background, QtGui.QBrush(qt_pixmap))
        self.ui.info_label.setAutoFillBackground(True)
        self.ui.info_label.setPalette(palette)

    def set_current_frame(self, new_current_frame, render = True):
        self.current_frame = new_current_frame
        self.message_queue.put(
            {'type': 'setText',
            'widget': 'cur_frame_label',
            'text': str(self.current_frame)}
        )
        self.info('Frame ' + str(self.current_frame))
        self.updateFramePositioner.emit()
        
        if render:
            self.stop_frame_rendering_thread()
            self.message_queue.put(
                {'type': 'setText',
                'widget': 'render_button',
                'text': 'Stop'}
            )

        if render:
            self.frame_thread = threading.Thread(target=self._process_current_frame, kwargs={'single_frame': True})
            self.frame_thread.daemon = True
            self.frame_thread.start()

    def create_temp_library(self, selection):        
        try:
            import flame

            clip = selection[0]
            temp_library_name = self.app_name + '_' + self.fw.sanitized(clip.name.get_value()) + '_' + self.fw.create_timestamp_uid()
            self.temp_library_name = temp_library_name
            self.temp_library = flame.projects.current_project.create_shared_library(temp_library_name)
            flame.execute_shortcut('Save Project')
            flame.projects.current_project.refresh_shared_libraries()
            return self.temp_library
        
        except Exception as e:
            message_string = f'Unable to create temp shared library:\n"{e}"'
            self.message_queue.put(
                {'type': 'mbox',
                'message': message_string,
                'action': self.close_application}
            )
            return None

    def get_total_channels_number(self, selection):
        num_channels = 0
        for clip in selection:
            clip_node_id = clip.get_wiretap_node_id()
            server_handle = WireTapServerHandle('localhost')
            clip_node_handle = WireTapNodeHandle(server_handle, clip_node_id)
            fmt = WireTapClipFormat()
            if not clip_node_handle.getClipFormat(fmt):
                message_string = f'Unable to obtain clip format: {clip_node_handle.lastError()}'
                self.message_queue.put(
                    {'type': 'mbox',
                    'message': message_string,
                    'action': None}
                )
            num_channels += fmt.numChannels()
        return num_channels
    
    def create_destination_node(self, clip, num_frames):
        try:
            import flame
            import numpy as np

            model_name = self.model_state_dict.get('model_name', 'UnknownModel')
            self.destination_node_name = clip.name.get_value() + f'_{model_name}_ML'
            destination_node_id = ''
            server_handle = WireTapServerHandle('localhost')
            clip_node_id = clip.get_wiretap_node_id()
            clip_node_handle = WireTapNodeHandle(server_handle, clip_node_id)
            fmt = WireTapClipFormat()
            if not clip_node_handle.getClipFormat(fmt):
                raise Exception('Unable to obtain clip format: %s.' % clip_node_handle.lastError())
            bits_per_channel = fmt.bitsPerPixel() // fmt.numChannels()
            self.bits_per_channel = bits_per_channel
            self.format_tag = fmt.formatTag()
            self.fmt = fmt

            self.temp_library.release_exclusive_access()
            node_id = self.temp_library.get_wiretap_node_id()
            parent_node_handle = WireTapNodeHandle(server_handle, node_id)
            destination_node_handle = WireTapNodeHandle()

            if not parent_node_handle.createClipNode(
                self.destination_node_name,  # display name
                fmt,  # clip format
                "CLIP",  # extended (server-specific) type
                destination_node_handle,  # created node returned here
            ):
                raise Exception(
                    "Unable to create clip node: %s." % parent_node_handle.lastError()
                )

            if not destination_node_handle.setNumFrames(int(num_frames)):
                raise Exception(
                    "Unable to set the number of frames: %s." % clip_node_handle.lastError()
                )
            
            dest_fmt = WireTapClipFormat()
            if not destination_node_handle.getClipFormat(dest_fmt):
                raise Exception(
                    "Unable to obtain clip format: %s." % clip_node_handle.lastError()
                )
            
            # '''
            metadata = dest_fmt.metaData()
            metadata_tag = dest_fmt.metaDataTag()
            metadata = metadata.replace('<ProxyFormat>default</ProxyFormat>', '<ProxyFormat>none</ProxyFormat>')
            destination_node_handle.setMetaData(metadata_tag, metadata)
            # '''

            destination_node_id = destination_node_handle.getNodeId().id()

        except Exception as e:
            message_string = f'Error creating destination wiretap node:\n {e}'
            self.message_queue.put(
                {'type': 'mbox',
                'message': message_string,
                'action': None}
            )
            return None
        finally:
            server_handle = None
            clip_node_handle = None
            parent_node_handle = None
            destination_node_handle = None

        return destination_node_id

    def delete_destination_node(self, destination_node_id):
        server_handle = WireTapServerHandle('localhost')
        clip_node_handle = WireTapNodeHandle(server_handle, destination_node_id)
        clip_node_handle.destroyNode()
        server_handle = None
        clip_node_handle = None

    def scan_models(self, folder_path):
        import importlib.util

        def import_model_from_file(file_path):
            module_name = os.path.splitext(os.path.basename(file_path))[0]
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return getattr(module, 'Model', None)

        model_dict = {}
        for filename in os.listdir(folder_path):
            if filename.endswith('.py'):
                file_path = os.path.join(folder_path, filename)
                ModelClass = import_model_from_file(file_path)
                if ModelClass:
                    try:
                        model_name = ModelClass.get_name()  # Assuming get_name() is the static method                            
                        model_dict[model_name] = ModelClass.get_model()
                    except Exception as e:
                        self.log(f'Error loading model from {file_path}: {e}')
        return model_dict

    def fill_model_menu(self):
        model_menu_items = self.prefs.get('recent_models')
        current_model_index = self.prefs.get('current_model_index', '99')
        if not isinstance(model_menu_items, dict):
            model_menu_items = {'99': 'Load Model ... '}
            self.prefs['recent_models'] = model_menu_items
            self.fw.save_prefs()
        if '99' not in model_menu_items.keys():
            model_menu_items['99'] = 'Load Model ... '
            self.prefs['recent_models'] = model_menu_items
            self.fw.save_prefs()

        model_menu = QtWidgets.QMenu(self)
        for model_number in sorted(model_menu_items.keys(), reverse=False):
            model_file_name = os.path.basename(model_menu_items.get(model_number, str()))
            model_name, _ = os.path.splitext(model_file_name)
            code = model_name
            action = model_menu.addAction(code)
            x = lambda chk=False, model_number=model_number: self.select_model(model_number)
            action.triggered[()].connect(x)
        self.message_queue.put(
            {'type': 'setMenu',
            'widget': 'model_selector',
            'menu': model_menu}
        )
        # self.ui.model_selector.setMenu(model_menu)
        current_model_path = model_menu_items.get(current_model_index)
        if not current_model_path:
            self.message_queue.put(
                {'type': 'setText',
                'widget': 'model_selector',
                'text': 'Load Model ... '}
            )
        else:
            current_model_filename = os.path.basename(current_model_path)
            current_model_name, _ = os.path.splitext(current_model_filename)
            self.message_queue.put(
                {'type': 'setText',
                'widget': 'model_selector',
                'text': current_model_name}
            )

    def select_model(self, model_number):
        import flame

        if model_number == '99': # load model code
            selected_model_dict_path = None
            self.hide()
            flame.browser.show(
                title = 'Select flameSimpleML Model:',
                extension = 'pth',
                default_path = self.prefs.get('model_state_dict_path', os.path.expanduser('~')),
                multi_selection = False)
            if len(flame.browser.selection) > 0:
                selected_model_dict_path = flame.browser.selection[0]
            self.show()
            self.raise_()
            self.activateWindow()
            self.processEvents()

            if not selected_model_dict_path:
                return False
            load_model_thread = threading.Thread(target=self._add_model_to_menu, args=(selected_model_dict_path,))
            load_model_thread.daemon = True
            load_model_thread.start()
        else:
            model_menu_items = self.prefs.get('recent_models')
            selected_model_dict_path = model_menu_items.get(model_number)
            if not selected_model_dict_path:
                return False
            if not self.load_model_state_dict(selected_model_dict_path):
                return False
            if not self.load_model(self.model_state_dict):
                return False
            self.prefs['current_model_index'] = model_number
            self.fw.save_prefs()
            current_model_filename = os.path.basename(selected_model_dict_path)
            current_model_name, _ = os.path.splitext(current_model_filename)
            self.message_queue.put(
                {'type': 'setText',
                'widget': 'model_selector',
                'text': current_model_name}
            )

        self.stop_frame_rendering_thread()        
        self.frame_thread = threading.Thread(target=self._process_current_frame, kwargs={'single_frame': True})
        self.frame_thread.daemon = True
        self.frame_thread.start()

    def _add_model_to_menu(self, selected_model_dict_path):
        if not self.load_model_state_dict(selected_model_dict_path):
            return False
        if not self.load_model(self.model_state_dict):
            return False
        self.add_model_to_menu(selected_model_dict_path)

    def add_model_to_menu(self, selected_model_dict_path):
        model_menu_items = self.prefs.get('recent_models')
        new_model_menu_items = {}
        if not isinstance(model_menu_items, dict):
            model_menu_items = {'99': 'Load Model ... '}

        for key in model_menu_items.keys():
            if key == '99':
                new_model_menu_items[key] = model_menu_items[key]
            else:
                try:
                    if int(key) + 1 < 10:
                        new_model_menu_items[f'{int(key) + 1}'] = model_menu_items[key]
                except:
                    pass
                
        new_model_menu_items['1'] = selected_model_dict_path
        self.prefs['recent_models'] = new_model_menu_items
        self.prefs['current_model_index'] = '1'
        self.fw.save_prefs()
        self.fill_model_menu()

    def load_model(self, model_state_dict):
        model_name = model_state_dict.get('model_name', 'MultiRes_v001')
        if model_name not in self.models.keys():
            message_string = f'Unable to load model {model_name} - unknown model name.\n'
            message_string += f'Known models are:\n{", ".join(self.models.keys())}'
            self.message_queue.put(
                {'type': 'mbox',
                'message': message_string,
                'action': None}
            )
            return False
        input_channles = model_state_dict.get('input_channels', 3)
        output_channels = model_state_dict.get('output_channels', 3)
        try:
            self.current_model = self.models[model_name](input_channles, output_channels).to(self.torch_device)
            self.current_model.half()
            self.current_model.load_state_dict(model_state_dict['model_state_dict'])
            self.current_model.half()
            self.current_model.eval()
            self.input_channels = input_channles
            self.output_channels = output_channels
            self.delete_destination_node(self.destination_node_id)
            self.message_queue.put({'type': 'info', 'message': 'Creating destination clip node...'})
            self.destination_node_id = self.create_destination_node(
                self.selection[0],
                self.selection[0].duration.frame
                )
            return True
        except Exception as e:
            message_string = f'Unable to load model state:\n{e}'
            self.message_queue.put(
                {'type': 'mbox',
                'message': message_string,
                'action': None}
            )
            return False

    def load_model_state_dict(self, selected_model_dict_path):
        import torch

        try:
            self.message_queue.put({'type': 'info', 'message': f'Loading model state dict {selected_model_dict_path}'})
            self.model_state_dict = torch.load(selected_model_dict_path)
            self.message_queue.put({'type': 'info', 'message': f'Model state dict loaded from {selected_model_dict_path}'})
            return True
        except Exception as e:
            message_string = f'Unable to load model state dict:\n{selected_model_dict_path}\n\n{e}'
            self.message_queue.put(
                {'type': 'mbox',
                'message': message_string,
                'action': None}
            )
            return False

    def process_messages(self):
        timeout = 1e-8

        while self.threads:
            try:
                item = self.message_queue.get_nowait()
            except queue.Empty:
                if not self.threads:
                    break
                time.sleep(timeout)
                continue
            if item is None:
                time.sleep(timeout)
                continue
            if not isinstance(item, dict):
                self.message_queue.task_done()
                time.sleep(timeout)
                continue

            item_type = item.get('type')

            if not item_type:
                self.message_queue.task_done()
                time.sleep(timeout)
                continue
            elif item_type == 'info':
                message = item.get('message')
                self._info(f'{message}')
            elif item_type == 'message':
                message = item.get('message')
                self._message(f'{message}')
            elif item_type == 'image':
                self.updateInterfaceImage.emit(item)
            elif item_type == 'setText':
                self.setText.emit(item)
            elif item_type == 'setMenu':
                self.setMenu.emit(item)
            elif item_type == 'mbox':
                self.showMessageBox.emit(item)
            else:
                self.message_queue.task_done()
                time.sleep(timeout)
                continue
            
            time.sleep(timeout)
        return

    def process_interface_images(self):
        timeout = 1e-8
        while self.threads:
            try:
                item = self.ui_images_queue.get_nowait()
            except queue.Empty:
                if not self.threads:
                    break
                time.sleep(timeout)
                continue
            if item is None:
                time.sleep(timeout)
                continue
            if not isinstance(item, dict):
                self.message_queue.task_done()
                time.sleep(timeout)
                continue

            item_type = item.get('type')

            if not item_type:
                self.message_queue.task_done()
                time.sleep(timeout)
                continue
            elif item_type == 'image':
                self.updateInterfaceImage.emit(item)
            else:
                self.message_queue.task_done()
                time.sleep(timeout)
                continue            
            time.sleep(timeout)
        return

    def process_frames_to_save(self):
        timeout = 1e-8

        while self.threads:
            try:
                item = self.frames_to_save_queue.get_nowait()
            except queue.Empty:
                if not self.threads:
                    break
                time.sleep(timeout)
                continue
            if item is None:
                time.sleep(timeout)
                continue
            if not isinstance(item, dict):
                self.message_queue.task_done()
                time.sleep(timeout)
                continue
            
            try:
                self._save_result_frame(
                    item.get('image_data'),
                    item.get('frame_number')
                )
                self.message_queue.task_done()
            except:
                time.sleep(timeout)
            
            time.sleep(timeout)
        return

    def info(self, message):
        item = {
            'type': 'info',
            'message': message
        }
        self.message_queue.put(item)

    def _info(self, message):
        try:
            self.ui.info_label.setText(str(message))
            QtWidgets.QApplication.instance().processEvents()
        except:
            pass

    def compose_frames_map(self, selection):
        pass

    def toggle_render_button(self):
        if self.ui.render_button.text() == 'Render':
            self.stop_frame_rendering_thread()
            self.current_state['rendering_by_render_button'] = True
            self.message_queue.put(
                    {'type': 'setText',
                    'widget': 'render_button',
                    'text': 'Stop'}
                )
            self.render_loop()
        else:
            self.stop_render_loop_thread()
            self.stop_frame_rendering_thread()
            self.current_state['rendering_by_render_button'] = False
            self.message_queue.put(
                    {'type': 'setText',
                    'widget': 'render_button',
                    'text': 'Render'}
                )
        return

        self.log_debug(f'render: self.rendering: {self.rendering}')
        self.rendering = not self.rendering
        button_text = 'Stop' if self.rendering else 'Render'
        self.message_queue.put(
                {'type': 'setText',
                'widget': 'render_button',
                'text': button_text}
            )

        self.processEvents()
        '''
        self.ui.render_button.setText(button_text)
        QtWidgets.QApplication.instance().processEvents()
        time.sleep(0.001)
        self.ui.render_button.setText(button_text)
        QtWidgets.QApplication.instance().processEvents()
        '''
        if self.rendering:
            self.render_loop()

    def render_loop(self):
        self.render_loop_thread = threading.Thread(target=self._render_loop)
        self.render_loop_thread.daemon = True
        self.render_loop_thread.start()

    def _render_loop(self):
        render_loop_start = time.time()
        print (f'min frame: {self.min_frame}, max frame: {self.max_frame}')
        for frame in range(self.min_frame, self.max_frame + 1):
            print (f'Frame = {frame}')
            if not self.threads:
                return
            if not self.current_state.get('rendering_by_render_button'):
                return
            # if self.frames_map[frame].get('saved'):
            #    self.info('Frame ' + str(self.current_frame) + ': Already saved')
            #    continue
            # self.set_current_frame(frame, render = True)
            '''
            if self.current_state.get('rendering_by_render_button'):
                save_image_data = self.current_state.get('res_image_data')
                self.save_result_frame(
                    save_image_data,
                    self.current_frame - 1
                )
                del save_image_data
            '''

            # self._process_current_frame()

        time_spent = time.time() - render_loop_start
        self.info(f'Rendering completed in {int(time_spent // 60)} min {int(time_spent % 60)} sec')
        self.rendering = False
        self.current_state['rendering_by_render_button'] = False
        self.message_queue.put(
                {'type': 'setText',
                'widget': 'render_button',
                'text': 'Render'}
            )
        self.processEvents()
        return

    def stop_frame_rendering_thread(self):
        if isinstance(self.frame_thread, threading.Thread):
            if self.frame_thread.is_alive():
                self.info(f'Frame {self.current_frame}: Stopping...')
                self.rendering = False
                self.frame_thread.join()

    def stop_render_loop_thread(self):
        if isinstance(self.render_loop_thread, threading.Thread):
            if self.render_loop_thread.is_alive():
                self.info(f'Frame {self.current_frame}: Stopping render loop thread ...')
                self.current_state['rendering_by_render_button'] = False
                self.render_loop_thread.join()

    def _process_current_frame(self, single_frame=False):
        timestamp = time.time()

        self.message_queue.put(
            {'type': 'info', 
            'message': f'Frame {self.current_frame}: reading source image(s) data ...'}
            )
        
        src_image_data = self.read_selection_data(
            self.selection, 
            self.current_frame - 1
            )
        
        self.current_state['src_image_data'] = src_image_data
        
        if self.current_model is None:
            self.update_interface_image_torch(
                src_image_data[:, :, :3],
                self.ui.image_res_label,
                text = 'Frame: ' + str(self.current_frame)
            )
            return
        
        if self.current_state.get('view_mode') == 'F1':
            if not self.current_state.get('rendering_by_render_button'):
                self.update_interface_image_torch(
                    src_image_data[:, :, :3],
                    self.ui.image_res_label,
                    text = 'Frame: ' + str(self.current_frame)
                )
                return
        

        if not self.current_state.get('rendering_by_render_button'):
            self.update_interface_image_torch(
                src_image_data[:, :, :3],
                self.ui.image_res_label,
                text = 'Frame: ' + str(self.current_frame)
            )

        '''
        print (f'before inference')
        # mem test report block
        import torch
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
        allocated_memory = torch.cuda.memory_allocated(self.torch_device)
        reserved_memory = torch.cuda.memory_reserved(self.torch_device)
        print(f"Allocated memory: {allocated_memory / 1e9:.2f} GB")
        print(f"Reserved memory:  {reserved_memory / 1e9:.2f} GB")
        print(f"Model size:  {sys.getsizeof(self.current_model) / 1e9:.2f} GB")
        # print(torch.cuda.memory_summary(device=None, abbreviated=False))
        '''


        try:
            res_image_data = self.apply_model(src_image_data)
        except Exception as e:
            res_image_data = None
            self.message_queue.put(
                {'type': 'info', 
                'message': f'Frame {self.current_frame}'}
                )
            message_string = f'Error processing frame:\n{e}'
            self.message_queue.put(
                {'type': 'mbox',
                'message': message_string,
                'action': None}
            )
            self.toggle_render_button()
            return
        
        if res_image_data is not None:
            self.update_interface_image_torch(
                    res_image_data[:, :, :3],
                    self.ui.image_res_label,
                    text = 'Frame: ' + str(self.current_frame)
                )

        self.current_state['res_image_data'] = res_image_data

        '''
        del res_image_data
        print (f'after inference')
        import torch
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
        allocated_memory = torch.cuda.memory_allocated(self.torch_device)
        reserved_memory = torch.cuda.memory_reserved(self.torch_device)
        print(f"Allocated memory: {allocated_memory / 1e9:.2f} GB")
        print(f"Reserved memory:  {reserved_memory / 1e9:.2f} GB")
        print(f"Model size:  {sys.getsizeof(self.current_model) / 1e9:.2f} GB")
        # print(torch.cuda.memory_summary(device=None, abbreviated=False))
        '''

    def apply_model(self, src_image_data):
        import torch
        from torch.nn import functional as F
     
        self.message_queue.put(
            {'type': 'info', 
            'message': f'Frame {self.current_frame}: Processing...'}
            )

        h, w, _ = src_image_data.shape
        ph = ((h - 1) // 64 + 1) * 64
        pw = ((w - 1) // 256 + 1) * 256
        padding = (0, pw - w, 0, ph - h)
        src_image_data = src_image_data.permute (2, 0, 1)
        src_image_data = F.pad(src_image_data, padding)
        src_image_data = src_image_data.unsqueeze(0)
        src_image_data = self.fw.normalize_values(src_image_data, torch = self.torch)
        src_image_data = src_image_data.to(self.torch_device, dtype=torch.half)        

        time_stamp = time.time()
        
        with torch.no_grad():
            output = self.current_model(src_image_data*2 -1)
        
        frame_time = time.time() - time_stamp
        frame_time_str = str(f'{frame_time:.2f}')
        print (f'time: {frame_time_str}')

        self.empty_torch_cache()

        rgb_output = (output[0] + 1) / 2
        rgb_output = self.fw.restore_normalized_values(rgb_output, torch = self.torch)
        rgb_output = rgb_output.permute(1, 2, 0)[:h, :w]

        result_image = rgb_output.to(dtype=torch.float32)

        self.message_queue.put(
            {'type': 'info', 
            'message': f'Frame {self.current_frame}'}
            )
        
        return result_image

    def read_selection_data(self, selection, frame_number):
        import torch

        tensors = []
        for clip in selection:
            clip_image_data = self.read_image_data_torch(clip, frame_number)
            tensors.append(clip_image_data)
        
        try:
            concatenated_data = torch.cat(tensors, dim=2)
            return concatenated_data
        except Exception as e:
            message_string = f'Unable to read source images data:\n"{e}"'
            self.message_queue.put(
                {'type': 'mbox',
                'message': message_string,
                'action': None}
            )
    
    def read_image_data_torch(self, clip, frame_number):
        import flame
        import numpy as np
        import torch

        try:
            server_handle = WireTapServerHandle('localhost')
            clip_node_id = clip.get_wiretap_node_id()
            clip_node_handle = WireTapNodeHandle(server_handle, clip_node_id)
            fmt = WireTapClipFormat()
            if not clip_node_handle.getClipFormat(fmt):
                raise Exception('Unable to obtain clip format: %s.' % clip_node_handle.lastError())
            num_frames = WireTapInt()
            if not clip_node_handle.getNumFrames(num_frames):
                raise Exception(
                    "Unable to obtain number of frames: %s." % clip_node_handle.lastError()
                )

            buff = "0" * fmt.frameBufferSize()

            if not clip_node_handle.readFrame(int(frame_number), buff, fmt.frameBufferSize()):
                raise Exception(
                    '[read_image_data] Unable to obtain read frame %i: %s.' % (frame_number, clip_node_handle.lastError())
                )
            
            frame_buffer_size = fmt.frameBufferSize()
            
            bits_per_channel = fmt.bitsPerPixel() // fmt.numChannels()

            if bits_per_channel == 8:
                buff_tail = frame_buffer_size - (fmt.height() * fmt.width() * fmt.numChannels())
                np_image_array = np.frombuffer(bytes(buff, 'latin-1'), dtype=np.uint8)[:-1 * buff_tail]
                image_array = torch.from_numpy(np_image_array.copy())
                del np_image_array
                image_array = image_array.to(
                    device = self.torch_device,
                    dtype = torch.float32,
                    non_blocking=True
                    )
                image_array = image_array.reshape((fmt.height(), fmt.width(),  fmt.numChannels()))
                image_array = torch.flip(image_array, [0])
                return image_array / 255

            elif bits_per_channel == 10:
                dt = np.uint16
                byte_array = np.frombuffer(bytes(buff, 'latin-1'), dtype='>u4')
                # byte_array = np.frombuffer(bytes(buff, 'latin-1'), dtype='<u4')
                values_10bit = np.empty((len(byte_array) * fmt.numChannels(),), dtype=np.uint16)
                values_10bit[::3] = (byte_array >> 22) & 0x3FF
                values_10bit[1::3] = (byte_array >> 12) & 0x3FF
                values_10bit[2::3] = (byte_array >> 2) & 0x3FF
                image_array = torch.from_numpy(values_10bit.astype(np.float32))
                image_array = image_array[:fmt.height() * fmt.width() * fmt.numChannels()]
                image_array = image_array.to(
                    device = self.torch_device,
                    dtype = torch.float32,
                    non_blocking=True
                    )
                image_array = image_array.reshape((fmt.height(), fmt.width(),  fmt.numChannels()))
                image_array = torch.flip(image_array, [0])
                return image_array / 1024

            elif bits_per_channel == 16 and not('float' in fmt.formatTag()):
                dt = np.uint16
                buff_tail = (frame_buffer_size // np.dtype(dt).itemsize) - (fmt.height() * fmt.width() * fmt.numChannels())
                image_array = np.frombuffer(bytes(buff, 'latin-1'), dtype=dt)[:-1 * buff_tail]
                image_array = torch.from_numpy(image_array.astype(np.float32))
                image_array = image_array.to(
                    device = self.parent_app.torch_device,
                    dtype = torch.float32,
                    non_blocking=True
                    )
                image_array = image_array.reshape((fmt.height(), fmt.width(),  fmt.numChannels()))
                image_array = torch.flip(image_array, [0])
                return image_array / 65535
            
            elif (bits_per_channel == 16) and ('float' in fmt.formatTag()):
                buff_tail = (frame_buffer_size // np.dtype(np.float16).itemsize) - (fmt.height() * fmt.width() * fmt.numChannels())
                np_image_array = np.frombuffer(bytes(buff, 'latin-1'), dtype=np.float16)[:-1 * buff_tail]
                image_array = torch.from_numpy(np_image_array.copy())
                del np_image_array
                image_array = image_array.to(
                    device = self.torch_device,
                    dtype = torch.float32,
                    non_blocking=True
                    )
                image_array = image_array.reshape((fmt.height(), fmt.width(),  fmt.numChannels()))
                image_array = torch.flip(image_array, [0])
                return image_array

            elif bits_per_channel == 32:
                buff_tail = (frame_buffer_size // np.dtype(np.float32).itemsize) - (fmt.height() * fmt.width() * fmt.numChannels())
                np_image_array = np.frombuffer(bytes(buff, 'latin-1'), dtype=np.float32)[:-1 * buff_tail]
                image_array = torch.from_numpy(np_image_array.copy())
                del np_image_array
                image_array = image_array.to(
                    device = self.torch_device,
                    dtype = torch.float32,
                    non_blocking=True
                    )
                image_array = image_array.reshape((fmt.height(), fmt.width(),  fmt.numChannels()))
                image_array = torch.flip(image_array, [0])
                return image_array

            else:
                raise Exception('Unknown image format')
            
        except Exception as e:
            self.message('Error: %s' % e)

        finally:
            server_handle = None
            clip_node_handle = None

    def save_result_frame(self, image_data, frame_number):
        self.frames_to_save_queue.put(
            {
                'image_data': image_data,
                'frame_number': frame_number
            }
        )

    def _save_result_frame(self, image_data, frame_number):
        import flame
        import numpy as np

        ext = '.exr' if 'float' in self.fmt.formatTag() else '.dpx'
            
        file_path = os.path.join(
            self.temp_folder,
            str(frame_number) + ext
        )

        save_file_start = time.time()

        try:
            if not os.path.isdir(os.path.dirname(file_path)):
                os.makedirs(os.path.dirname(file_path))

            height, width, depth = image_data.shape
            red = image_data[:, :, 0]
            green = image_data[:, :, 1]
            blue = image_data[:, :, 2]
            if depth > 3:
                alpha = image_data[:, :, 3]
            else:
                alpha = np.array([])

            if file_path.endswith('exr'):
                if self.bits_per_channel == 32:
                    self.parent_app.write_exr(
                        file_path,
                        width,
                        height,
                        red,
                        green,
                        blue,
                        alpha = alpha,
                        half_float = False
                    )
                else:
                    self.write_exr(
                        file_path,
                        width,
                        height,
                        red.astype(np.float16),
                        green.astype(np.float16),
                        blue.astype(np.float16),
                        alpha = alpha.astype(np.float16)
                    )

            else:
                self.write_dpx(
                    file_path,
                    width,
                    height,
                    red,
                    green,
                    blue,
                    alpha = alpha,
                    bit_depth = self.bits_per_channel
                )

            file_save_time = time.time() - save_file_start
            read_back_start = time.time()

            gateway_server_id = WireTapServerId('Gateway', 'localhost')
            gateway_server_handle = WireTapServerHandle(gateway_server_id)
            clip_node_handle = WireTapNodeHandle(gateway_server_handle, file_path + '@CLIP')
            fmt = WireTapClipFormat()
            if not clip_node_handle.getClipFormat(fmt):
                raise Exception('Unable to obtain clip format: %s.' % clip_node_handle.lastError())
            
            buff = "0" * fmt.frameBufferSize()

            if not clip_node_handle.readFrame(0, buff, fmt.frameBufferSize()):
                raise Exception(
                    'Unable to obtain read frame %i: %s.' % (frame_number, clip_node_handle.lastError())
                )
            
            read_back_time = time.time() - read_back_start
            framestore_write_start = time.time()

            server_handle = WireTapServerHandle('localhost')
            destination_node_handle = WireTapNodeHandle(server_handle, self.destination_node_id)
            dest_fmt = WireTapClipFormat()
            if not destination_node_handle.getClipFormat(dest_fmt):
                raise Exception('Unable to obtain clip format: %s.' % clip_node_handle.lastError())
            
            '''
            frame_id = WireTapStr()
            if not destination_node_handle. getFrameId(
                frame_number, frame_id
            ):
                raise Exception(
                    "Unable to obtain write frame %i: %s."
                    % (frame_number, destination_node_handle.lastError())
                )
            
            if not server_handle.writeFrame(
                frame_id, buff, dest_fmt.frameBufferSize()
            ):
                raise Exception(
                    "Unable to obtain write frame %i: %s."
                    % (frame_number, destination_node_handle.lastError())
                )

            '''

            num_children = WireTapInt(0)
            if not destination_node_handle.getNumChildren(num_children):
                raise Exception(
                    "Unable to obtain number of children: %s"
                    % destination_node_handle.lastError()
                )

            child = WireTapNodeHandle()
            child_name = WireTapStr()
            child_type_str = WireTapStr()
            for child_index in range(0, num_children):
                # Get the child node.
                #
                destination_node_handle.getChild(child_index, child)

                # Get the node's display name and type.
                #
                if not child.getDisplayName(child_name):
                    raise Exception(
                        "Unable to obtain node name: %s." % child.lastError()
                    )

                if not child.getNodeTypeStr(child_type_str):
                    raise Exception(
                        "Unable to obtain node type: %s." % child.lastError()
                    )
                
                if child_type_str.c_str() == 'LOWRES':
                    if not child.writeFrame(
                        frame_number, buff, dest_fmt.frameBufferSize()
                    ):
                        raise Exception(
                            "Unable to obtain write frame %i: %s."
                            % (frame_number, destination_node_handle.lastError())
                        )

                # Print the node info.
                #
                # print("Node: '%s' type: %s" % (child_name.c_str(), child_type_str.c_str()))

            if not destination_node_handle.writeFrame(
                frame_number, buff, dest_fmt.frameBufferSize()
            ):
                raise Exception(
                    "Unable to obtain write frame %i: %s."
                    % (frame_number, destination_node_handle.lastError())
                )
            
            framestore_write_time = time.time() - framestore_write_start

            self.log_debug(f'save file: {file_save_time:.2f}, read back: {read_back_time:.2f}, fs save: {framestore_write_time:.2f}')

            os.remove(file_path)

        except Exception as e:
            pprint (e)
            self.message('Error: %s' % e)
        finally:
            gateway_server_handle = None
            clip_node_handle = None
            server_handle = None
            destination_node_handle = None

        # flame.schedule_idle_event(wiretap_test)

    def write_exr(self, filename, width, height, red, green, blue, alpha, half_float = True, pixelAspectRatio = 1.0):
        import numpy as np
        import struct

        MAGIC = 20000630
        VERSION = 2
        UINT = 0
        HALF = 1
        FLOAT = 2

        channels_list = ['B', 'G', 'R'] if not alpha.size else ['A', 'B', 'G', 'R']

        def write_attr(f, name, type, value):
            f.write(name.encode('utf-8') + b'\x00')
            f.write(type.encode('utf-8') + b'\x00')
            f.write(struct.pack('<I', len(value)))
            f.write(value)

        def get_channels_attr(channels_list):
            channel_list = b''
            for channel_name in channels_list:
                name_padded = channel_name[:254] + '\x00'
                bit_depth = 1 if half_float else 2
                pLinear = 0
                reserved = (0, 0, 0)  # replace with your values if needed
                xSampling = 1  # replace with your value
                ySampling = 1  # replace with your value
                channel_list += struct.pack(
                    f"<{len(name_padded)}s i B 3B 2i",
                    name_padded.encode(), 
                    bit_depth, 
                    pLinear, 
                    *reserved, 
                    xSampling, 
                    ySampling
                    )
            channel_list += struct.pack('c', b'\x00')

                # channel_list += (f'{i}\x00').encode('utf-8')
                # channel_list += struct.pack("<i4B", HALF, 1, 1, 0, 0)
            return channel_list
        
        def get_box2i_attr(x_min, y_min, x_max, y_max):
            return struct.pack('<iiii', x_min, y_min, x_max, y_max)

        with open(filename, 'wb') as f:
            # Magic number and version field
            f.write(struct.pack('I', 20000630))  # Magic number
            f.write(struct.pack('H', 2))  # Version field
            f.write(struct.pack('H', 0))  # Version field
            write_attr(f, 'channels', 'chlist', get_channels_attr(channels_list))
            write_attr(f, 'compression', 'compression', b'\x00')  # no compression
            write_attr(f, 'dataWindow', 'box2i', get_box2i_attr(0, 0, width - 1, height - 1))
            write_attr(f, 'displayWindow', 'box2i', get_box2i_attr(0, 0, width - 1, height - 1))
            write_attr(f, 'lineOrder', 'lineOrder', b'\x00')  # increasing Y
            write_attr(f, 'pixelAspectRatio', 'float', struct.pack('<f', pixelAspectRatio))
            write_attr(f, 'screenWindowCenter', 'v2f', struct.pack('<ff', 0.0, 0.0))
            write_attr(f, 'screenWindowWidth', 'float', struct.pack('<f', 1.0))
            f.write(b'\x00')  # end of header

            # Scan line offset table size and position
            line_offset_pos = f.tell()
            pixel_data_start = line_offset_pos + 8 * height
            bytes_per_channel = 2 if half_float else 4
            # each scan line starts with 4 bytes for y coord and 4 bytes for pixel data size
            bytes_per_scan_line = width * len(channels_list) * bytes_per_channel + 8 

            for y in range(height):
                f.write(struct.pack('<Q', pixel_data_start + y * bytes_per_scan_line))

            channel_data = {'R': red, 'G': green, 'B': blue, 'A': alpha}

            # Pixel data
            for y in range(height):
                f.write(struct.pack('I', y))  # Line number
                f.write(struct.pack('I', bytes_per_channel * len(channels_list) * width))  # Pixel data size
                for channel in sorted(channels_list):
                    f.write(channel_data[channel][y].tobytes())
            f.close

    def write_dpx(self, filename, width, height, red, green, blue, alpha, bit_depth):
        import struct
        import numpy as np

        depth = 3 if not alpha.size else 4
        if bit_depth == 8:
            dt = np.uint8
            red = (red * 255).astype(dt)
            green = (green * 255).astype(dt)
            blue = (blue * 255).astype(dt)
            alpha = (alpha * 255).astype(dt)
        elif bit_depth == 16:
            dt = np.uint16
            red = (red * 65535).astype(dt)
            green = (green * 65535).astype(dt)
            blue = (blue * 65535).astype(dt)
            alpha = (alpha * 65535).astype(dt)
        else:
            dt = np.float32

        arr = np.ones((height, width, depth), dtype=dt)

        arr[:,:,0] = red
        arr[:,:,1] = green
        arr[:,:,2] = blue
        if alpha.size:
            arr[:,:,3] = alpha
            
        file_size = 8192 + arr.size * bit_depth // 8

        new_meta = {}
        new_meta['colorimetry'] = 4
        new_meta['copyright'] = '\x00' * 200
        new_meta['creator'] = '\x00' * 100
        new_meta['data_sign'] = 0
        new_meta['depth'] = bit_depth
        new_meta['descriptor'] = 50 if not alpha.size else 51
        new_meta['ditto'] = 1
        new_meta['dpx_version'] = 'V1.0\x00\x00\x00\x00'
        new_meta['encoding'] = 0
        new_meta['encryption_key'] = 4294967295
        new_meta['endianness'] = 'be'
        new_meta['file_size'] = file_size
        new_meta['filename'] = os.path.basename(filename) + '\x00' * (100 - len(filename))
        new_meta['height'] = height
        new_meta['image_element_count'] = 1
        new_meta['image_element_description'] = 'IMAGE DESCRIPTION DATA        \x00P'
        new_meta['image_padding'] = 0
        new_meta['input_device_name'] = '\x00' * 32
        new_meta['input_device_sn'] = '\x00' * 32
        new_meta['line_padding'] = 0
        new_meta['magic'] = 'SDPX'
        new_meta['offset'] = 8192
        new_meta['orientation'] = 0
        new_meta['packing'] = 0 if bit_depth != 10 else 1
        new_meta['project_name'] = '\x00' * 200
        new_meta['timestamp'] = '\x00' * 24
        new_meta['transfer_characteristic'] = 4
        new_meta['width'] = width

        propertymap = [
            #(field name, offset, length, type)

            ('magic', 0, 4, 'magic'),
            ('offset', 4, 4, 'I'),
            ('dpx_version', 8, 8, 'utf8'),
            ('file_size', 16, 4, 'I'),
            ('ditto', 20, 4, 'I'),
            ('filename', 36, 100, 'utf8'),
            ('timestamp', 136, 24, 'utf8'),
            ('creator', 160, 100, 'utf8'),
            ('project_name', 260, 200, 'utf8'),
            ('copyright', 460, 200, 'utf8'),
            ('encryption_key', 660, 4, 'I'),

            ('orientation', 768, 2, 'H'),
            ('image_element_count', 770, 2, 'H'),
            ('width', 772, 4, 'I'),
            ('height', 776, 4, 'I'),

            ('data_sign', 780, 4, 'I'),
            ('descriptor', 800, 1, 'B'),
            ('transfer_characteristic', 801, 1, 'B'),
            ('colorimetry', 802, 1, 'B'),
            ('depth', 803, 1, 'B'),
            ('packing', 804, 2, 'H'),
            ('encoding', 806, 2, 'H'),
            ('line_padding', 812, 4, 'I'),
            ('image_padding', 816, 4, 'I'),
            ('image_element_description', 820, 32, 'utf8'),

            ('input_device_name', 1556, 32, 'utf8'),
            ('input_device_sn', 1588, 32, 'utf8')
        ]

        def writeDPX(f, image, meta):
            endianness = ">" if meta['endianness'] == 'be' else "<"
            for p in propertymap:
                if p[0] in meta:
                    f.seek(p[1])
                    if p[3] == 'magic':
                        bytes = ('SDPX' if meta['endianness'] == 'be' else 'XPDS').encode(encoding='UTF-8')
                    elif p[3] == 'utf8':
                        bytes = meta[p[0]].encode(encoding='UTF-8')
                    else:
                        bytes = struct.pack(endianness + p[3], meta[p[0]])
                    f.write(bytes)
            if meta['depth'] == 10:
                raw = ((((image[:,:,0] * 0x000003FF).astype(np.dtype(np.int32)) & 0x000003FF) << 22) 
                        | (((image[:,:,1] * 0x000003FF).astype(np.dtype(np.int32)) & 0x000003FF) << 12)
                        | (((image[:,:,2] * 0x000003FF).astype(np.dtype(np.int32)) & 0x000003FF) << 2)
                    )
            else:
                raw = image.flatten()

            if meta['endianness'] == 'be':
                raw = raw.byteswap()

            f.seek(meta['offset'])
            raw.tofile(f, sep="")

        with open(filename, 'wb') as f:
            writeDPX(f, arr, new_meta)
            f.close()

    def empty_torch_cache(self):
        if sys.platform == 'darwin':
            try:
                import torch
                self.torch_device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            except:
                pass
        else:
            try:
                import torch
                if torch.cuda.is_available():
                    self.torch_device = torch.device('cuda')
                    torch.cuda.empty_cache()
            except:
                pass

    def set_device(self):
        if sys.platform == 'darwin':
            try:
                import torch
                torch_device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
                return torch_device
            except:
                return 'cpu'
        else:
            try:
                import torch
                torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                return torch_device
            except:
                return 'cpu'

    def close_application(self):
        import gc
        import flame

        self.stop_frame_rendering_thread()
        del self.current_state['src_image_data'], self.current_state['res_image_data']

        '''
        def print_all_tensors():
            print ('printing all tensors')
            for obj in gc.get_objects():
                try:
                    if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                        print(type(obj), obj.size())
                except:
                    pass

        print_all_tensors()
        '''

        '''
        try:
            for key in self.parent_app.current_models.keys():
                del self.self.parent_app.current_models[key]
            self.parent_app.current_models = {}
        except Exception as e:
            print (f'close_application exception {e}')
        '''

        del self.current_model
        del self.model_state_dict
        gc.collect()
        self.empty_torch_cache()

        while not self.frames_to_save_queue.empty():
            qsize = self.frames_to_save_queue.qsize()
            self.info(f'Waiting for {qsize} frames to be saved')
            time.sleep(0.01)
        
        result_clip = None
        if not self.temp_library:
            self.temp_library = None
            self.deleteLater()
            return False
        
        try:
            self.temp_library.acquire_exclusive_access()

            flame.execute_shortcut('Save Project')
            flame.execute_shortcut('Refresh Thumbnails')
            self.temp_library.commit()
            if self.destination_node_id:
                try:
                    result_clip = flame.find_by_wiretap_node_id(self.destination_node_id)
                except:
                    result_clip = None
            else:
                result_clip = None

            if not result_clip:
                # try harder
                flame.execute_shortcut('Save Project')
                flame.execute_shortcut('Refresh Thumbnails')
                self.temp_library.commit()
                ch = self.temp_library.children
                for c in ch:
                    if c.name.get_value() == self.destination_node_name:
                        result_clip = c
            
            if not result_clip:
                flame.execute_shortcut('Save Project')
                flame.execute_shortcut('Refresh Thumbnails')
                self.temp_library.commit()
                if self.destination_node_id:
                    try:
                        result_clip = flame.find_by_wiretap_node_id(self.destination_node_id)
                    except:
                        result_clip = None
                else:
                    result_clip = None

            if not result_clip:
                # try harder
                flame.execute_shortcut('Save Project')
                flame.execute_shortcut('Refresh Thumbnails')
                self.temp_library.commit()
                ch = self.temp_library.children
                for c in ch:
                    if c.name.get_value() == self.destination_node_name:
                        result_clip = c
            
            if result_clip:
                try:
                    copied_clip = flame.media_panel.copy(
                        source_entries = result_clip, destination = self.clip_parent
                        )
                    self.temp_library.acquire_exclusive_access()
                    flame.delete(self.temp_library)
                    '''
                    copied_clip = copied_clip[0]
                    segment = copied_clip.versions[0].tracks[0].segments[0]
                    segment.create_effect('Colour Mgmt')
                    copied_clip.render()
                    '''
                    flame.execute_shortcut('Save Project')
                    flame.execute_shortcut('Refresh Thumbnails')
                except:
                    pass
        except Exception as e:
            self.on_showMessageBox({'message': pformat(e)})

        self.threads = False
        self.deleteLater() # close Progress window after all events are processed

        '''
        def rescan_hooks():
            flame.execute_shortcut('Rescan Python Hooks')
        flame.schedule_idle_event(rescan_hooks)
        '''
