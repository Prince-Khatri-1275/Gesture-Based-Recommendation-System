import streamlit as st
import subprocess
import os
import signal

PID_FILE = os.path.join(os.path.dirname(__file__), "camera_pid.txt")  # To track the process

st.title("Live Camera Control")
script_path =  os.path.join(os.path.dirname(__file__), "test_file_2.py")

# Start button
if st.button("Start Camera"):
    if not os.path.exists(PID_FILE):  # Only if not already running
        proc = subprocess.Popen(["python", script_path])  # Run OpenCV script
        with open(PID_FILE, "w") as f:
            f.write(str(proc.pid))  # Save PID to stop it later
        st.success("Camera started!")
    else:
        st.warning("Camera is already running.")

# Stop button
if st.button("Stop Camera"):
    if os.path.exists(PID_FILE):
        with open(PID_FILE, "r") as f:
            pid = int(f.read())
        try:
            os.kill(pid, signal.SIGTERM)  # Terminate process using PID
            os.remove(PID_FILE)
            st.success("Camera stopped.")
        except Exception as e:
            st.error(f"Failed to stop camera: {e}")
    else:
        st.warning("No running camera process found.")
