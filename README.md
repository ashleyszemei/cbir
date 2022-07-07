# cbir

This CBIR is adapted from [CBIR](https://github.com/pochih/CBIR) by [@pochih](https://github.com/pochih).

## How to install

Open PyCharm and clone this repository.

Create a new Python interpreter using new Virtualenv environment. The virtual environment should be called "venv".

For reference, my folder path is C:\Users\User\PycharmProjects\cbir
and the path to my virtual environment is C:\Users\User\PycharmProjects\cbir\venv

Inside the project directory, activate the virtual environment.

        C:\Users\User\PycharmProjects\cbir> venv\scripts\activate

After the virtual environment has been activated, install the packages.

        (venv) C:\Users\User\PycharmProjects\cbir> pip install requirements.txt

After requirements have been installed, type the following command:

        (venv) C:\Users\User\PycharmProjects\cbir> python cbir.py

You should see this appear in your terminal:

        (venv) C:\Users\User\PycharmProjects\cbir>python cbir.py
        * Serving Flask app 'cbir' (lazy loading)
        * Environment: production
        WARNING: This is a development server. Do not use it in a production deployment.
        Use a production WSGI server instead.
        * Debug mode: on
        * Running on http://127.0.0.1:5000 (Press CTRL+C to quit)
        * Restarting with stat
        * Debugger is active!
        * Debugger PIN: xxx-xxx-xxx
 
Go to the link specified in the "Running on..." line.

The UI should now appear on the screen.


