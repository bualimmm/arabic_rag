import streamlit as st

st.set_page_config(layout="wide")  # Use wide layout for better RTL display
st.title("محادثة البوت العربي")
st.write("اطرح أسئلة حول مستنداتك وسيحاول البوت الإجابة بناءً على السياق المسترجع.")

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


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("اكتب سؤالك هنا"): # Removed "What is up?"
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = "لا أعرف"  # Arabic translation
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
