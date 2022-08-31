from django.urls import path
from . import views
urlpatterns = [
    path("",views.index),
    path("login",views.login),
    path("register",views.register),
    path("registerdata",views.registerdata),
    path("home",views.home),
    path("loginin",views.loginin),
    path("opencamera",views.opencamera),
    path("mask",views.mask),
    path("userprofile",views.userprofile),
    path("feedback",views.feedback),
    path("delete_account",views.delete_account),
    path("logout",views.logout),
    path("submitfeedback",views.submitfeedback),
    path("accountdel",views.accountdel),
]
