
document.addEventListener('DOMContentLoaded', function () {
    const confirmButton = document.querySelector('.logout-button.confirm');
    const cancelButton = document.querySelector('.logout-button.cancel');

    
    if (window.location.pathname.endsWith('logout.html')) {
        if (confirmButton) {
            confirmButton.addEventListener('click', function () {
                sessionStorage.removeItem('isLoggedIn');

                alert('You have been logged out successfully.');

                window.location.href = 'login.html';
            });
        }

        if (cancelButton) {
            cancelButton.addEventListener('click', function () {
                window.location.href = 'home.html';
            });
        }

        if (sessionStorage.getItem('isLoggedIn') !== 'true') {
            window.location.href = 'login.html';
        }
    }
});