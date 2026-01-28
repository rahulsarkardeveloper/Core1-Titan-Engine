use std::ffi::CStr;
use std::os::raw::c_char;
use std::ptr;
 
pub extern "C" fn titan_encode(input: *const c_char) -> *mut i32 {
    if input.is_null() {
        return ptr::null_mut();
    }
    
    let c_str = unsafe { CStr::from_ptr(input) };
    let text = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };

    let mut tokens: Vec<i32> = text
        .chars()
        .map(|c| c as i32) // সিম্পল ম্যাপিং (Unicode)
        .collect();

    tokens.push(-1);
    
    let ptr = tokens.as_mut_ptr();
    std::mem::forget(tokens); 
    
    ptr
}

#[no_mangle]
pub extern "C" fn titan_free_tokens(ptr: *mut i32) {

    if ptr.is_null() { return; }
    unsafe {
        
        let _ = Vec::from_raw_parts(ptr, 0, 0);     
    }
}
