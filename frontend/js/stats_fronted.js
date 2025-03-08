// Updated appointments.js for Dynamic Appointments and Rescheduling

document.getElementById("newAppointmentBtn").addEventListener("click", function () {
    document.getElementById("appointmentForm").classList.toggle("hidden");
});

function addAppointment() {
    let name = document.getElementById("patientName").value;
    let date = document.getElementById("appointmentDate").value;
    let reason = document.getElementById("reason").value;
    
    if (name && date && reason) {
        let appointmentList = document.querySelector(".appointment-list");
        let newAppointment = document.createElement("div");
        newAppointment.classList.add("appointment");
        newAppointment.innerHTML = `<p>${name} - <span class='date'>${date}</span></p>
            <button onclick="openReschedule('${name}')">Reschedule</button>`;
        appointmentList.appendChild(newAppointment);
        document.getElementById("appointmentForm").classList.add("hidden");
    }
}

let rescheduleTarget = null;
function openReschedule(name) {
    rescheduleTarget = document.querySelector(`.appointment p:contains('${name}') .date`);
    document.getElementById("rescheduleForm").classList.remove("hidden");
}

function rescheduleAppointment() {
    let newDate = document.getElementById("newDateTime").value;
    if (newDate && rescheduleTarget) {
        rescheduleTarget.textContent = newDate;
        document.getElementById("rescheduleForm").classList.add("hidden");
    }
}

// Chart.js Implementation for View Analytics
const ctx = document.getElementById('patientChart').getContext('2d');
const patientChart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: ['Week 1', 'Week 2', 'Week 3', 'Week 4'],
        datasets: [{
            label: 'Patient Visits',
            data: [10, 15, 8, 20],
            borderColor: 'blue',
            borderWidth: 2
        }]
    }
});
