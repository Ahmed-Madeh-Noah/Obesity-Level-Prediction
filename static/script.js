document.addEventListener("DOMContentLoaded", function () {
  const form = document.querySelector("form");
  const modal = document.getElementById("customModal");
  const confirmYes = document.getElementById("confirmYes");
  const confirmNo = document.getElementById("confirmNo");

  form.addEventListener("submit", function (e) {
    e.preventDefault(); // stop default submit
    modal.style.display = "block"; // show modal
  });

  confirmYes.addEventListener("click", function () {
    modal.style.display = "none";
    form.submit(); // manually submit the form
  });

  confirmNo.addEventListener("click", function () {
    modal.style.display = "none"; // just close modal
  });
});