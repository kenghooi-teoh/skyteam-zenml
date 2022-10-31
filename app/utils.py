import app as st


def add_logo():
    st.markdown(
        """
        <style>
            [data-testid="stSidebarNav"] {
                background-image: url(https://drive.google.com/uc?id=1FLmFALHI-0A182zaKOWxw0NLdQ4Jwv_6);
                background-repeat: no-repeat;
                padding-top: 120px;
                background-position: 20px 20px;
            }
            # [data-testid="stSidebarNav"]::before {
            #     content: "My Company Name";
            #     margin-left: 20px;
            #     margin-top: 20px;
            #     font-size: 30px;
            #     position: relative;
            #     top: 100px;
            # }
        </style>
        """,
        unsafe_allow_html=True,
    )