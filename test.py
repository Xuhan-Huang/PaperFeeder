import os
import resend

resend.api_key = 're_BcDUtRuz_69prGode2yojR5dRWysBVc73'

params: resend.Emails.SendParams = {
    "from": "Acme <onboarding@resend.dev>",
    "to": ["gaoxin23@m.fudan.edu.cn"],
    "subject": "hello world",
    "html": "<strong>it works!</strong>",
}

email = resend.Emails.send(params)
print(email)