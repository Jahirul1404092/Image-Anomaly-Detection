# About License Authentication Method

※ See [here (JP only)](https://chowagiken.sharepoint.com/processdevelopment/SitePages/%E6%88%90%E6%9E%9C%E7%89%A9%E3%83%A9%E3%82%A4%E3%82%BB%E3%83%B3%E3%82%B9%E8%AA%8D%E8%A8%BC%E3%81%AB%E3%81%A4%E3%81%84%E3%81%A6.aspx?source=https%3A%2F%2Fchowagiken.sharepoint.com%2Fprocessdevelopment%2FSitePages%2FForms%2FByAuthor.aspx) for basic policy.


# How to issue a Licens
1. Install PyArmor v7.4.4
    - Already installed in `dev` environment
2. PyArmor initial setup
    1. DL the license file `pyarmor-regfile-3106.zip` for "PyArmor (Visee)" registerd with 1Password
    2. Activate the paid license
        ```sh
        $ pyarmor register <regfile for Visee>
        $ pyarmor register
          Done if the authentication informaiton is printed
        ```
3. Issue a license with following expiration date
    - `license/<License name>/license.lic` will be generated
    ```sh
    $ pyarmor licenses <License name> \
        --expired YYYY-MM-DD
    ```
4. Provide `license/<License name>/license.lic` to the customer