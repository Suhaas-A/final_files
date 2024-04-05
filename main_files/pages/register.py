import datetime
import streamlit as st
from database import *
import random
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from streamlit_modal import Modal


def validate(email, subject):
    sender_email = "suhaas062010@gmail.com"
    sender_password = "ogkl cmnc rnnq rgzj"
    receiver_email = email

    # create message container - the correct MIME type is multipart/alternative.
    msg = MIMEMultipart('alternative')
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = f"{subject}"

    code = [1, 2, 3, 4, 5, 6, 7, 8, 9, 1]
    random.shuffle(code)

    str_code = ''
    for num in code:
        str_code += str(num)

    # create the body of the message (a plain-text and an HTML version).
    text = 'Enter the following code. If it was not you then please ignore the email.'
    html = f'''
        <html>
            <body>
                <p>
                    Hello, <br>
                    This is your code : <br>
                    <div style="font-size: 35px;">
                        <b><i>{int(str_code)}</i></b>
                    </div>
                </p>
            </body>
        </html>
        '''

    # Attach both plain-text and HTML versions of the message to the MIMEMultipart object.
    part1 = MIMEText(text, 'plain')
    part2 = MIMEText(html, 'html')
    msg.attach(part1)
    msg.attach(part2)

    # create SMTP session for sending the mail
    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
    except Exception as e:
        print("Error: Unable to send email.", e)

    return int(str_code)


st.set_page_config(initial_sidebar_state="collapsed")

session = Session()

st.title('Register')

no_sidebar_style = """
    <style>
        div[data-testid="stSidebarNav"] {display: none;}
    </style>
"""
st.markdown(no_sidebar_style, unsafe_allow_html=True)

with st.sidebar:
    st.divider()
    st.header('some name')
    st.divider()
    st.page_link(page='pages/login.py', label='login')
    st.page_link(page='pages/register.py', label='register')

st.divider()

with st.form(key='register_form'):
    username = st.text_input('Username: ')
    st.divider()

    password = st.text_input('Password: ', type='password')
    st.divider()

    email = st.text_input('Email: ')
    st.divider()

    submit = st.form_submit_button('register')

signin = st.button('Sign in to your account')
if signin:
    st.switch_page('pages/login.py')

if submit:
    if not username:
        st.error('Username field is required')
    elif not password:
        st.error('Password field is required')
    else:
        users = session.query(Users).all()
        usernames, emails = [], []

        for user in users:
            usernames.append(user.username)
            emails.append(user.email)

        if username in usernames:
            st.error('Username already exists')
        elif email in emails:
            st.error('Email already exists')
        else:
            code = validate(email, 'Email verification for some app')
            st.session_state['credentials'] = {
                'username': username,
                'password': password,
                'email': email,
                'code': code
            }
            st.session_state['show'] = True

if 'error' in st.session_state:
    st.error(st.session_state['error'])
    st.session_state.pop('error')

if 'show' in st.session_state:
    modal = Modal(key='verification', title='Verification')
    with modal.container():
        with st.form(key='ver_form'):
            code_inp = st.number_input('Code (check your emails): ')
            st.divider()
            credentials = st.session_state['credentials']
            if st.form_submit_button('submit'):
                if int(code_inp) == credentials['code']:
                    session.add(
                        Users(
                            username=credentials['username'],
                            password=credentials['password'],
                            email=credentials['email'],
                            create_date=datetime.datetime.now()
                        )
                    )
                    session.commit()
                    st.session_state.pop('show')
                    st.switch_page('pages/login.py')
                else:
                    st.session_state['error'] = 'The code is invalid'
                    st.session_state.pop('show')
                    st.switch_page('pages/register.py')
        if st.button("Close"):
            st.session_state.pop('show')
            st.switch_page('pages/register.py')