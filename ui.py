import asyncio

import streamlit as st

# Set page config as the first command
st.set_page_config(layout="wide")

from rag import get_answer, allowSelfSignedHttps

# Enable self-signed HTTPS certificates (if needed)
allowSelfSignedHttps(True)

# Set page config and title
st.title("خبير الميزانية العامة")
st.write("اطرح أسئلة حول بيان الميزانية العامة وسيحاول الخبير الإجابة.")

# Set text direction to RTL
st.markdown(
    """
<style>
body {
    direction: rtl;
}
.stTextInput > div > div > input {
    text-align: right;
}
</style>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("المصادقة")  # Authentication
    if password_input := st.text_input("أدخل كلمة المرور:", type="password"):
        if password_input == st.secrets["password"]:
            st.success("تمت المصادقة")  # Access granted!
            auth_status = True
        else:
            st.warning("كلمة المرور غير صحيحة. تم رفض المصادقة.")  # Incorrect password. Access denied.
            auth_status = False

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input
if (prompt := st.chat_input("اكتب سؤالك هنا")) and auth_status:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get answer from RAG pipeline
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = asyncio.run(get_answer(prompt))
        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
