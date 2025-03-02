document.addEventListener('DOMContentLoaded', function () {
    const loginForm = document.getElementById('login-form');

    if (loginForm) {
        loginForm.addEventListener('submit', function (e) {
            e.preventDefault();

            const username = document.getElementById('username').value;
            const password = document.getElementById('password').value;

            const validUsername = "admin";
            const validPassword = "password";

            if (username === validUsername && password === validPassword) {
                sessionStorage.setItem('isLoggedIn', 'true');

                window.location.href = 'home.html';
            } else {
                alert('Invalid username or password. Please try again.');
            }
        });
    }

    if (sessionStorage.getItem('isLoggedIn') === 'true') {
        if (window.location.pathname.endsWith('login.html')) {
            window.location.href = 'home.html';
        }
    } else {
        const protectedPages = ['index.html', 'home.html', 'appointments.html', 'logout.html'];
        if (protectedPages.some(page => window.location.pathname.endsWith(page))) {
            window.location.href = 'login.html';
        }
    }
});